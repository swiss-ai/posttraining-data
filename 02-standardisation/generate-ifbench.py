import sys
import uuid
import json
import time
import random
import asyncio
import argparse
from typing import Any
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field

import aiohttp
import requests
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_from_disk

import nltk
nltk.download('punkt_tab')


import ifevalg.instructions_registry as instructions_registry

INSTRUCTION_DICT = instructions_registry.INSTRUCTION_DICT

# Global counter for tracking active sglang requests
current_requests = 0
INSTRUCTION_KEYS = list(INSTRUCTION_DICT.keys())
INSTRUCTION_CONFLICTS = instructions_registry.INSTRUCTION_CONFLICTS


@dataclass
class InstructionInstance:
    instruction_id: str
    description: str
    kwargs: dict[str, Any]

    def check_following(self, response: str) -> bool:
        try:
            instruction_cls = INSTRUCTION_DICT[self.instruction_id]
            instruction_obj = instruction_cls(self.instruction_id)

            if self.kwargs:
                instruction_obj.build_description(**self.kwargs)
            else:
                instruction_obj.build_description()

            return instruction_obj.check_following(response)
        except Exception as e:
            print(f"Verification failed for {self.instruction_id}: {e}")
            return False


@dataclass
class InstructionGroup:
    instructions: list[InstructionInstance] = field(default_factory=list)

    @property
    def descriptions(self) -> list[str]:
        return [inst.description for inst in self.instructions]

    @property
    def instruction_ids(self) -> list[str]:
        return [inst.instruction_id for inst in self.instructions]

    @property
    def all_kwargs(self) -> list[dict[str, Any]]:
        return [inst.kwargs for inst in self.instructions]

    def verify_response(self, response: str) -> tuple[bool, list[dict[str, Any]]]:
        results = []
        all_passed = True

        for inst in self.instructions:
            passed = inst.check_following(response)
            results.append(
                {
                    "instruction_id": inst.instruction_id,
                    "passed": passed,
                }
            )
            if not passed:
                all_passed = False

        return all_passed, results


class AsyncSGLangClient:
    def __init__(
        self,
        selection_url: str,
        generation_url: str,
        semaphore: asyncio.Semaphore,
        selection_tokenizer: AutoTokenizer,
        generation_tokenizer: AutoTokenizer,
        timeout: int = 180,
        max_response_tokens: int = 2048,
        reasoning_parser: str | None = "deepseek-r1",
        selection_model_name: str = "unknown",
        generation_model_name: str = "unknown",
    ):
        self.selection_url = selection_url.rstrip("/")
        self.generation_url = generation_url.rstrip("/")
        self.semaphore = semaphore
        self.selection_tokenizer = selection_tokenizer
        self.generation_tokenizer = generation_tokenizer
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_response_tokens = max_response_tokens
        self.reasoning_parser = reasoning_parser
        self.selection_model_name = selection_model_name
        self.generation_model_name = generation_model_name
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "AsyncSGLangClient":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session:
            await self._session.close()

    async def select_best_group(
        self,
        prompt: str,
        groups: list[InstructionGroup],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> tuple[int, int]:
        global current_requests
        selection_prompt = self._build_selection_prompt(prompt, groups)

        async with self.semaphore:
            current_requests += 1
            try:
                async with self._session.post(
                    f"{self.selection_url}/generate",
                    json={
                        "text": selection_prompt,
                        "sampling_params": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "regex": r"-1|[0-4]",
                        },
                    },
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    text = result["text"].strip()
                    completion_tokens = result["meta_info"]["completion_tokens"]
                    try:
                        selection = int(text)
                        if selection < -1 or selection > 4:
                            return (-1, completion_tokens)
                        return (selection, completion_tokens)
                    except ValueError:
                        return (-1, completion_tokens)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # print(f"SGLang selection request failed: {e}")
                return (-1, 0)
            finally:
                current_requests -= 1

    async def generate_response(
        self,
        augmented_prompt: str,
        original_response: str | None,
        instruction_descriptions: list[str],
        temperature: float = 0.7,
    ) -> tuple[str | None, int, str]:
        global current_requests
        generation_prompt = self._build_generation_prompt(
            augmented_prompt, original_response, instruction_descriptions
        )
        # print(f"[Generation] Using model: {self.generation_model_name}")

        request_json = {
            "text": generation_prompt,
            "sampling_params": {
                "max_new_tokens": self.max_response_tokens,
                "temperature": temperature,
                "skip_special_tokens": True,
            },
        }
        
        # Add reasoning_parser if specified (for automatic reasoning extraction)
        if self.reasoning_parser and self.reasoning_parser != "apriel":
            request_json["reasoning_parser"] = self.reasoning_parser

        async with self.semaphore:
            current_requests += 1
            try:
                async with self._session.post(
                    f"{self.generation_url}/generate",
                    json=request_json,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    raw_text = result["text"].strip() if result else None
                    completion_tokens = result["meta_info"]["completion_tokens"]
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # print(f"SGLang generation request failed: {e}")
                return (None, 0, self.generation_model_name)
            finally:
                current_requests -= 1

        if raw_text is None or self.reasoning_parser is None:
            return (raw_text, completion_tokens, self.generation_model_name)

        # For apriel, manually parse the reasoning tags
        if self.reasoning_parser == "apriel":
            clean_text = raw_text.split("</thinking>")[-1].split("<|end|>")[0].strip()
            return (clean_text, completion_tokens, self.generation_model_name)

        # For other parsers, check if reasoning extraction actually worked
        # Common markers that indicate reasoning is still present
        reasoning_markers = ["<think>", "<thinking>", "[REASONING]", "Let me think", "First,"]
        has_reasoning = any(marker.lower() in raw_text.lower()[:500] for marker in reasoning_markers)
        
        if not has_reasoning:
            # Reasoning was already extracted successfully
            return (raw_text, completion_tokens, self.generation_model_name)
        
        # Fallback: use selection model to extract just the final answer
        print(f"Warning: Automatic reasoning extraction failed, using fallback extraction...")
        try:
            extraction_prompt = self._build_reasoning_extraction_prompt(raw_text)
            async with self._session.post(
                f"{self.selection_url}/generate",
                json={
                    "text": extraction_prompt,
                    "sampling_params": {
                        "max_new_tokens": len(raw_text) + 100,
                        "temperature": 0.0,
                    },
                },
            ) as extract_response:
                extract_response.raise_for_status()
                extract_result = await extract_response.json()
                clean_text = extract_result["text"].strip() if extract_result else raw_text
                return (clean_text, completion_tokens, self.generation_model_name)
        except Exception as e:
            print(f"Fallback reasoning extraction failed: {e}, using raw response")
            return (raw_text, completion_tokens, self.generation_model_name)

    def _build_selection_prompt(
        self, prompt: str, groups: list[InstructionGroup]
    ) -> str:
        groups_text = ""
        for i, group in enumerate(groups):
            instructions_text = "\n".join(f"  - {desc}" for desc in group.descriptions)
            groups_text += f"\nGroup {i}:\n{instructions_text}\n"

        user_message = f"""You are evaluating which group of instructions is most compatible with a given user prompt. 
The instructions should be feasible to follow when answering the prompt.

User Prompt: {prompt}

Available Instruction Groups:{groups_text}

Select the group number (0-4) that contains instructions most compatible with the prompt.
If none of the groups are suitable (e.g., instructions conflict with the prompt's nature), output -1.

Your selection (output only the number):"""

        messages = [{"role": "user", "content": user_message}]
        formatted_prompt = self.selection_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted_prompt

    def _build_generation_prompt(
        self,
        augmented_prompt: str,
        original_response: str | None,
        instruction_descriptions: list[str],
    ) -> str:
        instructions_list = "\n".join(f"- {desc}" for desc in instruction_descriptions)

        if original_response:
            user_message = f"""Answer the following question while strictly following ALL the formatting instructions listed below.

Question: {augmented_prompt}

You MUST follow these instructions in your response:
{instructions_list}

Here is a reference answer that you should use as inspiration for the content (but you must adapt it to follow all the instructions above):
{original_response}

Now provide your response that follows all the instructions:"""
        else:
            user_message = f"""Answer the following question while strictly following ALL the formatting instructions listed below.

Question: {augmented_prompt}

You MUST follow these instructions in your response:
{instructions_list}

Provide your response:"""

        messages = [{"role": "user", "content": user_message}]
        formatted_prompt = self.generation_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted_prompt

    def _build_reasoning_extraction_prompt(self, response_with_reasoning: str) -> str:
        """Build a prompt to extract just the final answer from a response that contains reasoning."""
        user_message = f"""The following text contains reasoning followed by a final answer. Extract ONLY the final answer, removing all reasoning, thinking process, or intermediate steps.

Text with reasoning:
{response_with_reasoning}

Extract only the final answer:"""
        
        messages = [{"role": "user", "content": user_message}]
        formatted_prompt = self.selection_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted_prompt


class MultiAsyncSGLangClient:
    """Wrapper that randomly selects from multiple AsyncSGLangClient instances."""

    def __init__(
        self,
        clients: list[AsyncSGLangClient],
    ):
        self.clients = clients
        if not self.clients:
            raise ValueError("At least one client must be provided")

    async def __aenter__(self) -> "MultiAsyncSGLangClient":
        # Enter all client contexts
        for client in self.clients:
            await client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Exit all client contexts
        for client in self.clients:
            await client.__aexit__(exc_type, exc_val, exc_tb)

    def _get_random_client(self) -> AsyncSGLangClient:
        """Randomly select a client from the pool."""
        return random.choice(self.clients)

    async def select_best_group(
        self,
        prompt: str,
        groups: list[InstructionGroup],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> tuple[int, int]:
        """Select best group using a randomly chosen client."""
        client = self._get_random_client()
        return await client.select_best_group(
            prompt, groups, max_new_tokens, temperature
        )

    async def generate_response(
        self,
        augmented_prompt: str,
        original_response: str | None,
        instruction_descriptions: list[str],
        temperature: float = 0.7,
    ) -> tuple[str | None, int, str]:
        """Generate response using a randomly chosen client."""
        client = self._get_random_client()
        return await client.generate_response(
            augmented_prompt, original_response, instruction_descriptions, temperature
        )


def has_conflict(instruction_id: str, existing_ids: set[str]) -> bool:
    if instruction_id in INSTRUCTION_CONFLICTS:
        conflicts = INSTRUCTION_CONFLICTS[instruction_id]
        if existing_ids & conflicts:
            return True
    return False


def build_instruction_instance(
    instruction_id: str, prompt: str
) -> InstructionInstance | None:
    instruction_cls = INSTRUCTION_DICT[instruction_id]
    instruction_obj = instruction_cls(instruction_id)

    PROMPT_TO_REPEAT_INSTRUCTIONS = {
        "combination:repeat_prompt",
        "copy:copy",
        "copy:copying_simple",
        "copy:copying_multiple",
        "new:copy_span_idx",
    }

    try:
        if instruction_id in PROMPT_TO_REPEAT_INSTRUCTIONS:
            if not prompt or len(prompt.strip()) == 0:
                return None
            description = instruction_obj.build_description(prompt_to_repeat=prompt)
        elif instruction_id == "keywords:exclude_word_harder":
            if not prompt or len(prompt.strip()) == 0:
                return None
            description = instruction_obj.build_description(instruction=prompt)
        else:
            description = instruction_obj.build_description()

        kwargs = instruction_obj.get_instruction_args()
        if kwargs is None:
            kwargs = {}

        return InstructionInstance(
            instruction_id=instruction_id,
            description=description,
            kwargs=kwargs,
        )

    except Exception as e:
        print(f"Failed to build instruction {instruction_id}: {e}")
        return None


def sample_instruction_group(
    prompt: str, num_instructions: int, max_attempts: int = 100
) -> InstructionGroup | None:
    group = InstructionGroup()
    existing_ids: set[str] = set()
    attempts = 0

    while len(group.instructions) < num_instructions and attempts < max_attempts:
        attempts += 1

        instruction_id = random.choice(INSTRUCTION_KEYS)

        if instruction_id in existing_ids:
            continue
        if has_conflict(instruction_id, existing_ids):
            continue

        instance = build_instruction_instance(instruction_id, prompt)
        if instance is None:
            continue

        group.instructions.append(instance)
        existing_ids.add(instruction_id)

    if len(group.instructions) < num_instructions:
        return None

    return group


def sample_multiple_groups(
    prompt: str, num_instructions: int, num_groups: int = 5
) -> list[InstructionGroup]:
    groups = []
    for _ in range(num_groups):
        group = sample_instruction_group(prompt, num_instructions)
        if group is not None:
            groups.append(group)
    return groups


async def augment_single_turn_instruction(
    sample: dict[str, Any],
    sglang_client: AsyncSGLangClient | MultiAsyncSGLangClient,
    max_retries: int = 3,
    max_generation_retries: int = 3,
) -> dict[str, Any] | None:
    initial_prompt = sample.get("initial_prompt", {})
    if isinstance(initial_prompt, str):
        prompt = initial_prompt
    else:
        prompt = initial_prompt.get("content", "")

    if not prompt or len(prompt.strip()) == 0:
        return None

    original_response = None
    branches = sample.get("conversation_branches", [])
    if branches and len(branches) > 0:
        first_branch = branches[0]
        messages = first_branch.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                parts = msg.get("parts", [])
                for part in parts:
                    if part.get("type") == "response":
                        original_response = part.get("content", "")
                        break
                if original_response:
                    break

    num_instructions = random.randint(1, 5)

    selected_group = None
    total_selection_tokens = 0
    for _ in range(max_retries):
        groups = sample_multiple_groups(prompt, num_instructions, num_groups=5)
        if len(groups) == 0:
            continue

        selection, tokens = await sglang_client.select_best_group(prompt, groups)
        total_selection_tokens += tokens

        if selection >= 0 and selection < len(groups):
            selected_group = groups[selection]
            break

    if selected_group is None:
        return None

    instruction_descriptions = " ".join(selected_group.descriptions)
    augmented_prompt = f"{prompt} {instruction_descriptions}"

    generated_response = None
    verification_results = None
    all_passed = False
    total_generation_tokens = 0
    generation_model_used = "unknown"

    for _ in range(max_generation_retries):
        response_text, tokens, model_name = await sglang_client.generate_response(
            augmented_prompt=augmented_prompt,
            original_response=original_response,
            instruction_descriptions=selected_group.descriptions,
        )
        total_generation_tokens += tokens
        generation_model_used = model_name

        if response_text is None:
            continue

        generated_response = response_text
        all_passed, verification_results = selected_group.verify_response(
            generated_response
        )

        if all_passed:
            break

    if generated_response is None:
        return None

    ground_truth = {
        "instruction_id": selected_group.instruction_ids,
        "kwargs": selected_group.all_kwargs,
    }

    original_messages = {
        "initial_prompt": sample.get("initial_prompt", {}),
        "conversation_branches": sample.get("conversation_branches", []),
    }

    augmented_sample = {
        "conversation_id": f"ifbench-{uuid.uuid4().hex[:12]}",
        "dataset_source": "ifbench-augmented",
        "augmentation_type": "single_turn_instruction",
        "original_metadata": {
            "original_id": sample.get("conversation_id", ""),
            "original_source": sample.get("dataset_source", ""),
            "original_messages": json.dumps(original_messages),
            "generation_model": generation_model_used,
        },
        "system_prompt": sample.get("system_prompt", {"content": "", "metadata": {}}),
        "generation_model": generation_model_used,
        "initial_prompt": {
            "role": "user",
            "content": augmented_prompt,
            "metadata": {},
        },
        "available_functions": sample.get("available_functions", []),
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "response",
                                "content": generated_response,
                                "metadata": {},
                                "name": "",
                                "args": "",
                            }
                        ],
                    }
                ]
            }
        ],
        "ground_truth": json.dumps(ground_truth),
        "num_instructions": num_instructions,
        "verification_passed": all_passed,
        "verification_results": json.dumps(verification_results)
        if verification_results
        else None,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return augmented_sample


async def augment_single_turn_refinement(
    sample: dict[str, Any],
    sglang_client: AsyncSGLangClient | MultiAsyncSGLangClient,
    max_retries: int = 3,
    max_generation_retries: int = 3,
) -> dict[str, Any] | None:
    initial_prompt = sample.get("initial_prompt", {})
    if isinstance(initial_prompt, str):
        prompt = initial_prompt
    else:
        prompt = initial_prompt.get("content", "")

    if not prompt or len(prompt.strip()) == 0:
        return None

    branches = sample.get("conversation_branches", [])
    if not branches or len(branches) == 0:
        return None

    first_branch = branches[0]
    messages = first_branch.get("messages", [])
    if not messages:
        return None

    assistant_response = None
    for msg in messages:
        if msg.get("role") == "assistant":
            parts = msg.get("parts", [])
            for part in parts:
                if part.get("type") == "response":
                    assistant_response = part.get("content", "")
                    break
            if assistant_response:
                break

    if not assistant_response:
        return None

    num_instructions = random.randint(1, 3)

    selected_group = None
    total_selection_tokens = 0
    for _ in range(max_retries):
        groups = sample_multiple_groups(prompt, num_instructions, num_groups=5)
        if len(groups) == 0:
            continue

        selection, tokens = await sglang_client.select_best_group(prompt, groups)
        total_selection_tokens += tokens

        if selection >= 0 and selection < len(groups):
            selected_group = groups[selection]
            break

    if selected_group is None:
        return None

    instruction_descriptions = " ".join(selected_group.descriptions)

    refinement_prompt_templates = [
        "Please refine your previous response.",
        "Please revise your previous response.",
        "Please improve your previous response.",
        "Please update your previous response.",
        "Please enhance your previous response.",
        "Please polish your previous response.",
        "Please adjust your previous response.",
        "Please modify your previous response.",
        "Please rework your previous response.",
        "Please reconsider your previous response.",
        "Please refine the response you provided earlier.",
        "Please revise the response you provided earlier.",
        "Please improve the response you provided earlier.",
        "Please update the response you provided earlier.",
        "Please enhance the response you provided earlier.",
        "Could you make your previous response more detailed?",
        "Could you clarify your previous response?",
        "Could you make your previous response more concise?",
        "Could you elaborate on your previous response?",
        "Could you provide more specifics in your previous response?",
        "Could you add more examples to your previous response?",
        "Could you simplify your previous response?",
        "Could you be more precise in your previous response?",
        "I'd like you to refine what you said earlier.",
        "Can you improve upon your last response?",
        "Let's refine your previous answer.",
        "Your previous response needs some refinement.",
        "Can you do better with your previous response?",
        "Try to enhance what you wrote before.",
        "Let me see an improved version of your previous response.",
        "Go back and refine your earlier response.",
        "Can you polish that up a bit?",
        "Let's make that response better.",
        "How about improving that response?",
        "Can we refine that a bit more?",
        "Let's try that again with more refinement.",
        "Can you give me a better version of that?",
        "I would appreciate a refinement of your previous response.",
        "Kindly refine your earlier response.",
        "Please provide a refined version of your previous answer.",
        "I request that you enhance your previous response.",
        "Would you mind improving your earlier response?",
        "Take another pass at your previous response.",
        "Refine what you just provided.",
        "Improve on your last answer.",
        "Polish up that previous response.",
        "Make your previous response better.",
        "Work on improving your earlier response.",
        "Revamp your previous response.",
        "Make your previous response clearer and more accurate.",
        "Add more depth to your previous response.",
        "Strengthen your previous response.",
        "Make your previous response more comprehensive.",
        "Expand and refine your previous response.",
        "Correct and improve your previous response.",
    ]

    selected_template = random.choice(refinement_prompt_templates)
    refinement_prompt = f"{selected_template} {instruction_descriptions}"

    generated_response = None
    verification_results = None
    all_passed = False
    total_generation_tokens = 0
    generation_model_used = "unknown"

    for _ in range(max_generation_retries):
        response_text, tokens, model_name = await sglang_client.generate_response(
            augmented_prompt=refinement_prompt,
            original_response=assistant_response,
            instruction_descriptions=selected_group.descriptions,
        )
        total_generation_tokens += tokens
        generation_model_used = model_name

        if response_text is None:
            continue

        generated_response = response_text
        all_passed, verification_results = selected_group.verify_response(
            generated_response
        )

        if all_passed:
            break

    if generated_response is None:
        return None

    ground_truth = {
        "instruction_id": selected_group.instruction_ids,
        "kwargs": selected_group.all_kwargs,
    }

    original_messages = {
        "initial_prompt": sample.get("initial_prompt", {}),
        "conversation_branches": sample.get("conversation_branches", []),
    }

    augmented_sample = {
        "conversation_id": f"ifbench-{uuid.uuid4().hex[:12]}",
        "dataset_source": "ifbench-augmented",
        "augmentation_type": "single_turn_refinement",
        "original_metadata": {
            "original_id": sample.get("conversation_id", ""),
            "original_source": sample.get("dataset_source", ""),
            "original_messages": json.dumps(original_messages),
            "generation_model": generation_model_used,
        },
        "system_prompt": sample.get("system_prompt", {"content": "", "metadata": {}}),
        "initial_prompt": {
            "role": "user",
            "content": prompt,
            "metadata": {},
        },
        "available_functions": sample.get("available_functions", []),
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "response",
                                "content": assistant_response,
                                "metadata": {},
                                "name": "",
                                "args": "",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "response",
                                "content": refinement_prompt,
                                "metadata": {},
                                "name": "",
                                "args": "",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "response",
                                "content": generated_response,
                                "metadata": {},
                                "name": "",
                                "args": "",
                            }
                        ],
                    },
                ]
            }
        ],
        "ground_truth": json.dumps(ground_truth),
        "num_instructions": num_instructions,
        "verification_passed": all_passed,
        "verification_results": json.dumps(verification_results)
        if verification_results
        else None,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_model": generation_model_used,
    }

    return augmented_sample


async def augment_multi_turn_instruction(
    sample: dict[str, Any],
    sglang_client: AsyncSGLangClient | MultiAsyncSGLangClient,
    max_retries: int = 3,
    max_generation_retries: int = 3,
) -> dict[str, Any] | None:
    branches = sample.get("conversation_branches", [])
    if not branches or len(branches) == 0:
        return None

    first_branch = branches[0]
    messages = first_branch.get("messages", [])

    user_turns = 0
    initial_prompt = sample.get("initial_prompt", {})
    if isinstance(initial_prompt, str):
        initial_content = initial_prompt
    else:
        initial_content = initial_prompt.get("content", "")

    if initial_content:
        user_turns += 1

    for msg in messages:
        if msg.get("role") == "user":
            user_turns += 1

    if user_turns < 2:
        return None

    original_first_response = None
    for msg in messages:
        if msg.get("role") == "assistant":
            parts = msg.get("parts", [])
            for part in parts:
                if part.get("type") == "response":
                    original_first_response = part.get("content", "")
                    break
            if original_first_response:
                break

    all_ground_truths = []
    all_verification_results = []
    augmented_messages = []
    initial_group = None

    total_selection_time = 0.0
    total_generation_time = 0.0
    total_selection_attempts = 0
    total_generation_attempts = 0
    total_selection_tokens = 0
    total_generation_tokens = 0

    if initial_content:
        num_instructions = random.randint(1, 3)
        selected_group = None

        # Track selection timing for initial turn
        selection_start = time.time()
        for attempt in range(max_retries):
            groups = sample_multiple_groups(
                initial_content, num_instructions, num_groups=5
            )
            if len(groups) == 0:
                continue
            total_selection_attempts += 1
            selection, tokens = await sglang_client.select_best_group(
                initial_content, groups
            )
            total_selection_tokens += tokens
            if selection >= 0 and selection < len(groups):
                selected_group = groups[selection]
                break
        total_selection_time += time.time() - selection_start

        if selected_group is None:
            return None

        initial_group = selected_group
        instruction_descriptions = " ".join(selected_group.descriptions)
        augmented_initial = f"{initial_content} {instruction_descriptions}"

        # Track generation timing for initial turn
        generation_start = time.time()
        generated_response = None
        verification_results = None
        all_passed = False
        generation_model_used = "unknown"

        for gen_attempt in range(max_generation_retries):
            total_generation_attempts += 1
            response_text, tokens, model_name = await sglang_client.generate_response(
                augmented_prompt=augmented_initial,
                original_response=original_first_response,
                instruction_descriptions=selected_group.descriptions,
            )
            total_generation_tokens += tokens
            generation_model_used = model_name

            if response_text is None:
                continue

            generated_response = response_text
            all_passed, verification_results = selected_group.verify_response(
                generated_response
            )

            if all_passed:
                break
        total_generation_time += time.time() - generation_start

        if generated_response is None:
            return None

        augmented_messages.append(
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "response",
                        "content": generated_response,
                        "metadata": {},
                        "name": "",
                        "args": "",
                    }
                ],
            }
        )

        all_ground_truths.append(
            {
                "turn": 0,
                "instruction_id": selected_group.instruction_ids,
                "kwargs": selected_group.all_kwargs,
            }
        )

        all_verification_results.append(
            {
                "turn": 0,
                "passed": all_passed,
                "details": verification_results,
            }
        )
    else:
        augmented_initial = initial_content

    turn_idx = 1
    prev_assistant_response = (
        augmented_messages[0]["parts"][0]["content"] if augmented_messages else None
    )

    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            parts = msg.get("parts", [])
            user_content = ""
            for part in parts:
                if part.get("type") == "response":
                    user_content = part.get("content", "")
                    break

            if user_content:
                num_instructions = random.randint(1, 3)
                selected_group = None

                # Track selection timing for this turn
                selection_start = time.time()
                for attempt in range(max_retries):
                    groups = sample_multiple_groups(
                        user_content, num_instructions, num_groups=5
                    )
                    if len(groups) == 0:
                        continue
                    total_selection_attempts += 1
                    selection, tokens = await sglang_client.select_best_group(
                        user_content, groups
                    )
                    total_selection_tokens += tokens
                    if selection >= 0 and selection < len(groups):
                        selected_group = groups[selection]
                        break
                total_selection_time += time.time() - selection_start

                if selected_group is None:
                    augmented_messages.append(msg)
                else:
                    instruction_descriptions = " ".join(selected_group.descriptions)
                    augmented_content = f"{user_content} {instruction_descriptions}"

                    augmented_messages.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "type": "response",
                                    "content": augmented_content,
                                    "metadata": {},
                                    "name": "",
                                    "args": "",
                                }
                            ],
                        }
                    )

                    original_response_for_turn = None
                    for j in range(i + 1, len(messages)):
                        if messages[j].get("role") == "assistant":
                            parts = messages[j].get("parts", [])
                            for part in parts:
                                if part.get("type") == "response":
                                    original_response_for_turn = part.get("content", "")
                                    break
                            break

                    # Track generation timing for this turn
                    generation_start = time.time()
                    generated_response = None
                    verification_results = None
                    all_passed = False

                    for gen_attempt in range(max_generation_retries):
                        total_generation_attempts += 1
                        response_text, tokens, model_name = await sglang_client.generate_response(
                            augmented_prompt=augmented_content,
                            original_response=original_response_for_turn,
                            instruction_descriptions=selected_group.descriptions,
                        )
                        total_generation_tokens += tokens
                        generation_model_used = model_name

                        if response_text is None:
                            continue

                        generated_response = response_text
                        all_passed, verification_results = (
                            selected_group.verify_response(generated_response)
                        )

                        if all_passed:
                            break
                    total_generation_time += time.time() - generation_start

                    if generated_response:
                        augmented_messages.append(
                            {
                                "role": "assistant",
                                "parts": [
                                    {
                                        "type": "response",
                                        "content": generated_response,
                                        "metadata": {},
                                        "name": "",
                                        "args": "",
                                    }
                                ],
                            }
                        )

                        all_verification_results.append(
                            {
                                "turn": turn_idx,
                                "passed": all_passed,
                                "details": verification_results,
                            }
                        )

                    all_ground_truths.append(
                        {
                            "turn": turn_idx,
                            "instruction_id": selected_group.instruction_ids,
                            "kwargs": selected_group.all_kwargs,
                        }
                    )
            else:
                augmented_messages.append(msg)
            turn_idx += 1

    if len(all_ground_truths) == 0:
        return None

    original_messages = {
        "initial_prompt": sample.get("initial_prompt", {}),
        "conversation_branches": sample.get("conversation_branches", []),
    }

    overall_passed = (
        all(v["passed"] for v in all_verification_results)
        if all_verification_results
        else False
    )

    augmented_sample = {
        "conversation_id": f"ifbench-{uuid.uuid4().hex[:12]}",
        "dataset_source": "ifbench-augmented",
        "augmentation_type": "multi_turn_instruction",
        "original_metadata": {
            "original_id": sample.get("conversation_id", ""),
            "original_source": sample.get("dataset_source", ""),
            "original_messages": json.dumps(original_messages),
            "generation_model": generation_model_used,
        },
        "system_prompt": sample.get("system_prompt", {"content": "", "metadata": {}}),
        "generation_model": generation_model_used,
        "initial_prompt": {
            "role": "user",
            "content": augmented_initial,
            "metadata": {},
        },
        "available_functions": sample.get("available_functions", []),
        "conversation_branches": [{"messages": augmented_messages}],
        "ground_truth": json.dumps(all_ground_truths),
        "num_instructions": sum(len(gt["instruction_id"]) for gt in all_ground_truths),
        "verification_passed": overall_passed,
        "verification_results": json.dumps(all_verification_results),
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return augmented_sample


@dataclass
class AugmentationTask:
    sample: dict[str, Any]
    aug_type: str


@dataclass
class AugmentationResult:
    aug_type: str
    result: dict[str, Any] | None
    success: bool


async def process_single_task(
    task: AugmentationTask,
    sglang_client: AsyncSGLangClient | MultiAsyncSGLangClient,
    augmentation_funcs: dict,
    max_retries: int,
    max_generation_retries: int,
) -> AugmentationResult:
    aug_func = augmentation_funcs.get(task.aug_type)
    if aug_func is None:
        return AugmentationResult(
            aug_type=task.aug_type,
            result=None,
            success=False,
        )

    try:
        result = await aug_func(
            task.sample, sglang_client, max_retries, max_generation_retries
        )
        return AugmentationResult(
            aug_type=task.aug_type,
            result=result,
            success=result is not None,
        )
    except Exception as e:
        print(f"Error processing task: {e}")
        return AugmentationResult(
            aug_type=task.aug_type,
            result=None,
            success=False,
        )


async def process_batch(
    tasks: list[AugmentationTask],
    sglang_client: AsyncSGLangClient | MultiAsyncSGLangClient,
    augmentation_funcs: dict,
    max_retries: int,
    max_generation_retries: int,
) -> list[AugmentationResult]:
    coroutines = [
        process_single_task(
            task, sglang_client, augmentation_funcs, max_retries, max_generation_retries
        )
        for task in tasks
    ]
    return await asyncio.gather(*coroutines)


def serialize_metadata(obj: Any) -> Any:
    """Recursively serialize all keys containing 'metadata' to JSON strings."""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if "metadata" in key.lower() and value is not None:
                result[key] = json.dumps(value)
            else:
                result[key] = serialize_metadata(value)
        return result
    elif isinstance(obj, list):
        return [serialize_metadata(item) for item in obj]
    else:
        return obj


def save_dataset(augmented_samples: list[dict[str, Any]], output_path: str) -> None:
    if not augmented_samples:
        return

    # Serialize metadata fields to avoid schema conflicts
    serialized_samples = [serialize_metadata(sample) for sample in augmented_samples]

    # Convert to Dataset and save as DatasetDict with train split
    dataset = Dataset.from_list(serialized_samples)
    dataset_dict = DatasetDict({"train": dataset})

    dataset_dict.save_to_disk(output_path)
    print(f"Saved dataset with {len(dataset)} examples to {output_path}")


async def process_dataset_async(
    input_path: str,
    output_path: str,
    selection_urls: list[str],
    generation_urls: list[str],
    selection_model_paths: list[str],
    generation_model_paths: list[str],
    reasoning_parsers: list[str] | None,
    target_samples: int | None,
    augmentation_types: list[str],
    max_retries: int,
    max_generation_retries: int,
    max_response_tokens: int,
    seed: int,
    concurrency: int,
    batch_size: int,
    save_interval: int = 100,
) -> None:
    random.seed(seed)

    num_selection_models = len(selection_model_paths)
    num_generation_models = len(generation_model_paths)

    if len(selection_urls) != num_selection_models:
        raise ValueError(
            f"Number of selection URLs ({len(selection_urls)}) "
            f"must match number of selection model paths ({num_selection_models})"
        )
    if len(generation_urls) != num_generation_models:
        raise ValueError(
            f"Number of generation URLs ({len(generation_urls)}) "
            f"must match number of generation model paths ({num_generation_models})"
        )

    print(
        f"Loading tokenizers: {num_selection_models} selection model(s), "
        f"{num_generation_models} generation model(s)..."
    )
    selection_tokenizers = []
    generation_tokenizers = []

    for i, sel_path in enumerate(selection_model_paths):
        print(
            f"  [selection {i + 1}/{num_selection_models}] Loading tokenizer from {sel_path}..."
        )
        sel_tok = AutoTokenizer.from_pretrained(sel_path, trust_remote_code=True)
        selection_tokenizers.append(sel_tok)

    for i, gen_path in enumerate(generation_model_paths):
        print(
            f"  [generation {i + 1}/{num_generation_models}] Loading tokenizer from {gen_path}..."
        )
        gen_tok = AutoTokenizer.from_pretrained(gen_path, trust_remote_code=True)
        generation_tokenizers.append(gen_tok)

    print(f"Loading dataset from {input_path}...")
    try:
        dataset = load_from_disk(input_path)
    except (TypeError, ValueError) as e:
        print(f"Error loading dataset with features schema: {e}")
        print("Attempting to load dataset by reconstructing from arrow files...")
        
        # The dataset_info.json has a corrupted schema. We need to load the raw arrow tables
        # and let the datasets library infer the schema automatically.
        import glob
        from datasets import Dataset as HFDataset
        
        # Find data arrow files (not cache files)
        arrow_pattern = f"{input_path}/data-*.arrow"
        arrow_files = sorted(glob.glob(arrow_pattern))
        
        if not arrow_files:
            # Try train split subdirectory
            arrow_pattern = f"{input_path}/train/data-*.arrow"
            arrow_files = sorted(glob.glob(arrow_pattern))
        
        if not arrow_files:
            raise RuntimeError(f"Cannot load dataset from {input_path}. No data arrow files found.") from e
        
        print(f"Found {len(arrow_files)} arrow file(s), loading and concatenating...")
        try:
            # These are Arrow IPC Stream files, not Arrow File format
            import pyarrow as pa
            tables = []
            for arrow_file in arrow_files:
                with pa.memory_map(arrow_file, 'r') as source:
                    # Use open_stream instead of open_file for IPC stream format
                    reader = pa.ipc.open_stream(source)
                    table = reader.read_all()
                    tables.append(table)
            
            # Concatenate all tables
            combined_table = pa.concat_tables(tables)
            
            # Strip the corrupted schema metadata from the table
            # This removes the embedded HuggingFace features metadata that's causing the error
            clean_schema = pa.schema(
                [pa.field(field.name, field.type) for field in combined_table.schema],
                metadata=None
            )
            combined_table = combined_table.cast(clean_schema)
            
            # Create dataset from the table without corrupted metadata
            dataset = HFDataset(combined_table)
            print(f"Successfully loaded {len(dataset)} examples from arrow files")
        except Exception as e2:
            print(f"Failed to load dataset from arrow files: {e2}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Cannot load dataset from {input_path}. Error: {e2}") from e

    if hasattr(dataset, "keys"):
        split_name = list(dataset.keys())[0]
        print(f"Using split: {split_name}")
        dataset = dataset[split_name]

    dataset = dataset.shuffle(seed=seed)

    dataset_size = len(dataset)
    if target_samples is not None:
        print(
            f"Target: {target_samples} output samples from {dataset_size} input samples (will loop if needed)"
        )
    else:
        print(f"Processing {dataset_size} samples (single pass)")

    semaphore = asyncio.Semaphore(concurrency)

    # Create multiple clients (cartesian product of selection and generation URLs/models)
    clients = []
    for sel_idx in range(num_selection_models):
        for gen_idx in range(num_generation_models):
            reasoning_parser = None
            if reasoning_parsers is not None:
                reasoning_parser = (
                    reasoning_parsers[gen_idx]
                    if gen_idx < len(reasoning_parsers)
                    else None
                )
            client = AsyncSGLangClient(
                selection_urls[sel_idx],
                generation_urls[gen_idx],
                semaphore,
                selection_tokenizers[sel_idx],
                generation_tokenizers[gen_idx],
                max_response_tokens=max_response_tokens,
                reasoning_parser=reasoning_parser,
                selection_model_name=selection_model_paths[sel_idx],
                generation_model_name=generation_model_paths[gen_idx],
            )
            clients.append(client)

    # Wrap clients in MultiAsyncSGLangClient for random selection
    multi_client = MultiAsyncSGLangClient(clients)
    print(
        f"Created {len(clients)} client(s) (cartesian selection x generation); requests will be randomly distributed across them."
    )

    augmentation_funcs = {
        "single_turn_instruction": augment_single_turn_instruction,
        "single_turn_refinement": augment_single_turn_refinement,
        "multi_turn_instruction": augment_multi_turn_instruction,
    }

    def create_tasks_for_sample(sample: dict[str, Any]) -> list[AugmentationTask]:
        tasks = []
        for aug_type in augmentation_types:
            if aug_type not in augmentation_funcs:
                continue

            if aug_type == "multi_turn_instruction":
                branches = sample.get("conversation_branches", [])
                if not branches or len(branches) == 0:
                    continue
                messages = branches[0].get("messages", [])
                user_count = sum(1 for m in messages if m.get("role") == "user")
                if user_count < 1:
                    continue

            tasks.append(AugmentationTask(sample=sample, aug_type=aug_type))
        return tasks

    augmented_samples = []

    stats = {aug_type: {"success": 0, "failed": 0} for aug_type in augmentation_types}
    pass_num = 0
    sample_idx = 0
    last_save_count = len(augmented_samples)

    async with multi_client as sglang_client:
        processed_count = 0  # Track tasks processed in this run
        initial_successful = len(augmented_samples)  # Existing successful samples
        if target_samples is not None:
            pbar = tqdm_asyncio(
                total=target_samples,
                initial=initial_successful,
                desc=f"reqs={current_requests} | Generating samples ({initial_successful} existing, {processed_count} processed)",
            )
        else:
            pbar = tqdm_asyncio(
                total=dataset_size * len(augmentation_types),
                initial=0,
                desc=f"reqs={current_requests} | Processing samples ({processed_count} processed)",
            )

        while True:
            # Check if we've reached the target
            if target_samples is not None and len(augmented_samples) >= target_samples:
                break

            # Check if we should stop (no target = single pass)
            if target_samples is None and sample_idx >= dataset_size:
                break

            # Build a batch of tasks
            tasks: list[AugmentationTask] = []
            while len(tasks) < batch_size:
                if (
                    target_samples is not None
                    and len(augmented_samples) + len(tasks) >= target_samples
                ):
                    break
                if target_samples is None and sample_idx >= dataset_size:
                    break

                # Get sample (loop through dataset if needed)
                actual_idx = sample_idx % dataset_size
                if actual_idx == 0 and sample_idx > 0:
                    pass_num += 1
                    print(f"\nStarting pass {pass_num + 1} through dataset...")

                sample = dataset[actual_idx]
                if hasattr(sample, "to_dict"):
                    sample = sample.to_dict()
                elif not isinstance(sample, dict):
                    sample = dict(sample)

                sample_tasks = create_tasks_for_sample(sample)
                tasks.extend(sample_tasks)
                sample_idx += 1

            if not tasks:
                break

            # Process tasks and update progress bar as each completes
            task_futures = [
                asyncio.create_task(
                    process_single_task(
                        task,
                        sglang_client,
                        augmentation_funcs,
                        max_retries,
                        max_generation_retries,
                    )
                )
                for task in tasks
            ]

            # Process results as they complete (not waiting for entire batch)
            for future in asyncio.as_completed(task_futures):
                try:
                    result = await future
                except asyncio.CancelledError:
                    continue

                if result.success and result.result is not None:
                    # Check if we've already reached target
                    if (
                        target_samples is not None
                        and len(augmented_samples) >= target_samples
                    ):
                        # Cancel remaining tasks
                        for remaining_future in task_futures:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    augmented_samples.append(result.result)
                    stats[result.aug_type]["success"] += 1
                    processed_count += 1
                    pbar.update(1)
                    current_successful = len(augmented_samples)
                    pbar.set_description(
                        f"reqs={current_requests} | Generating samples ({current_successful} successful, {processed_count} processed)"
                        if target_samples is not None
                        else f"reqs={current_requests} | Processing samples ({processed_count} processed)"
                    )

                    # Periodic save
                    if save_interval > 0:
                        new_samples_count = len(augmented_samples) - last_save_count
                        if new_samples_count >= save_interval:
                            print(
                                f"\nSaving checkpoint: {len(augmented_samples)} total samples..."
                            )
                            save_dataset(augmented_samples, output_path)
                            last_save_count = len(augmented_samples)
                else:
                    stats[result.aug_type]["failed"] += 1
                    processed_count += 1
                    # Update description for failed tasks too
                    current_successful = len(augmented_samples)
                    pbar.set_description(
                        f"reqs={current_requests} | Generating samples ({current_successful} successful, {processed_count} processed)"
                        if target_samples is not None
                        else f"reqs={current_requests} | Processing samples ({processed_count} processed)"
                    )
                    # Only update progress bar value if no target (since target is about successful samples)
                    if target_samples is None:
                        pbar.update(1)

                # Check if we should stop processing remaining tasks
                if (
                    target_samples is not None
                    and len(augmented_samples) >= target_samples
                ):
                    # Cancel remaining tasks
                    for remaining_future in task_futures:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    break

        pbar.close()

    print(f"\nCompleted after {pass_num + 1} pass(es) through the dataset")
    print("\nAugmentation Statistics:")
    for aug_type, counts in stats.items():
        total = counts["success"] + counts["failed"]
        if total > 0:
            success_rate = counts["success"] / total * 100
            print(
                f"  {aug_type}: {counts['success']}/{total} ({success_rate:.1f}% success)"
            )

    # Final save (only if there are new samples since last save)
    if len(augmented_samples) > last_save_count:
        print(f"\nPerforming final save: {len(augmented_samples)} total samples...")
        save_dataset(augmented_samples, output_path)
    elif augmented_samples:
        print(f"\nDataset already saved with {len(augmented_samples)} samples.")
    else:
        print("No samples were successfully augmented.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verifiable instruction-following dataset using IFBench"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input standardised dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for augmented dataset",
    )
    parser.add_argument(
        "--urls",
        type=str,
        default=None,
        help="Comma-separated list of SGLang server URLs (in same order as --model-paths). If provided, requests will be randomly distributed across these URLs. Must be used with --model-paths.",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        default=None,
        help="Comma-separated list of model paths. If provided, requests will be randomly distributed across these models. Must be used with --urls.",
    )
    parser.add_argument(
        "--selection-model-paths",
        type=str,
        default=None,
        help="Comma-separated list of model paths used only for selection (overrides --model-paths for selection).",
    )
    parser.add_argument(
        "--generation-model-paths",
        type=str,
        default=None,
        help="Comma-separated list of model paths used only for generation (overrides --model-paths for generation).",
    )
    parser.add_argument(
        "--reasoning-parsers",
        type=str,
        default=None,
        help="Comma-separated list of reasoning parser IDs (one per model). Default: deepseek-r1 for all models.",
    )
    parser.add_argument(
        "--selection-urls",
        type=str,
        default=None,
        help="Comma-separated list of SGLang URLs used only for selection (overrides --urls for selection).",
    )
    parser.add_argument(
        "--generation-urls",
        type=str,
        default=None,
        help="Comma-separated list of SGLang URLs used only for generation (overrides --urls for generation).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Target number of output samples to generate. If input dataset is smaller, multiple passes will be performed (default: one pass through input)",
    )
    parser.add_argument(
        "--augmentation-types",
        type=str,
        default="single_turn_instruction,single_turn_refinement,multi_turn_instruction",
        help="Comma-separated list of augmentation types",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum LLM retry attempts for group selection (default: 3)",
    )
    parser.add_argument(
        "--max-generation-retries",
        type=int,
        default=3,
        help="Maximum retries for response generation/verification (default: 3)",
    )
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for generated responses (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Maximum concurrent requests to SGLang (default: 32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of tasks per batch (default: 100)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save dataset every N new samples (default: 1000). Set to 0 to disable periodic saves.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def parse_csv(value: str | None) -> list[str] | None:
        if value is None:
            return None
        values = [v.strip() for v in value.split(",") if v.strip()]
        return values if values else None

    selection_model_paths = parse_csv(args.selection_model_paths) or parse_csv(
        args.model_paths
    )
    generation_model_paths = parse_csv(args.generation_model_paths) or parse_csv(
        args.model_paths
    )
    selection_urls = parse_csv(args.selection_urls) or parse_csv(args.urls)
    generation_urls = parse_csv(args.generation_urls) or parse_csv(args.urls)
    reasoning_parsers = parse_csv(args.reasoning_parsers)

    def normalize_parser(p: str | None) -> str | None:
        if p is None:
            return None
        lowered = p.strip().lower()
        if lowered in {"", "none", "null", "no", "skip", "off"}:
            return None
        return p.strip()

    if reasoning_parsers is not None:
        reasoning_parsers = [normalize_parser(p) for p in reasoning_parsers]

    if (
        selection_model_paths is None
        or generation_model_paths is None
        or selection_urls is None
        or generation_urls is None
    ):
        print(
            "Error: You must provide model paths and URLs. "
            "Use --model-paths/--urls for shared lists or "
            "--selection-model-paths/--generation-model-paths and "
            "--selection-urls/--generation-urls for separate lists."
        )
        sys.exit(1)

    if not selection_model_paths or not generation_model_paths:
        print(
            "Error: At least one selection and one generation model path must be provided"
        )
        sys.exit(1)
    if not selection_urls or not generation_urls:
        print("Error: At least one selection and one generation URL must be provided")
        sys.exit(1)

    if len(selection_model_paths) != len(selection_urls):
        print(
            f"Error: Number of selection model paths ({len(selection_model_paths)}) "
            f"must match number of selection URLs ({len(selection_urls)})"
        )
        sys.exit(1)
    if len(generation_model_paths) != len(generation_urls):
        print(
            f"Error: Number of generation model paths ({len(generation_model_paths)}) "
            f"must match number of generation URLs ({len(generation_urls)})"
        )
        sys.exit(1)
    if reasoning_parsers is not None and len(reasoning_parsers) not in (
        1,
        len(generation_model_paths),
    ):
        print(
            f"Error: Number of reasoning parsers ({len(reasoning_parsers)}) must be 1 or match number of generation models ({len(generation_model_paths)})."
        )
        sys.exit(1)
    if (
        reasoning_parsers is not None
        and len(reasoning_parsers) == 1
        and len(generation_model_paths) > 1
    ):
        reasoning_parsers = reasoning_parsers * len(generation_model_paths)

    print(
        "Multi-model mode: "
        f"{len(selection_model_paths)} selection model(s) and "
        f"{len(generation_model_paths)} generation model(s); "
        "requests will be randomly dispatched across clients built from the cartesian product of selection and generation URLs/models."
    )

    # Before processing, query /workers for each unique URL (selection and generation).
    def print_workers(urls: list[str], label: str) -> None:
        print(f"\n=== {label} workers ===")
        for url in urls:
            endpoint = f"{url.rstrip('/')}/workers"
            try:
                resp = requests.get(endpoint, timeout=5)
                resp.raise_for_status()
                print(f"[{url}] {resp.text}")
            except Exception as e:
                print(f"[{url}] Failed to fetch workers: {e}")

    # Preserve order while deduplicating URLs
    def unique_in_order(items: list[str]) -> list[str]:
        seen = set()
        ordered = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    print_workers(unique_in_order(selection_urls), "Selection")
    print_workers(unique_in_order(generation_urls), "Generation")

    augmentation_types = [t.strip() for t in args.augmentation_types.split(",")]

    valid_types = {
        "single_turn_instruction",
        "single_turn_refinement",
        "multi_turn_instruction",
    }
    for aug_type in augmentation_types:
        if aug_type not in valid_types:
            print(
                f"Warning: Unknown augmentation type '{aug_type}'. Valid types: {valid_types}"
            )

    asyncio.run(
        process_dataset_async(
            input_path=args.input,
            output_path=args.output,
            selection_urls=selection_urls,
            generation_urls=generation_urls,
            selection_model_paths=selection_model_paths,
            generation_model_paths=generation_model_paths,
            reasoning_parsers=reasoning_parsers,
            target_samples=args.num_samples,
            augmentation_types=augmentation_types,
            max_retries=args.max_retries,
            max_generation_retries=args.max_generation_retries,
            max_response_tokens=args.max_response_tokens,
            seed=args.seed,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
        )
    )


if __name__ == "__main__":
    main()
