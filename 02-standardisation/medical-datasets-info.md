# Medical Datasets - Pre-processed Parquet Overview

All parquet files are located at:

    /capstor/store/cscs/swissai/infra01/medical_text/pre-processed/

Source datasets are cached at:

    /capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache/

Conversion scripts are in 02-standardisation/convert-medical-*.py.

All output parquets share a common "messages" column (list of {role, content} dicts)
plus dataset-specific metadata columns.


## Dataset Summary

    Parquet File                          HF Dataset                                              Revision (short)   Rows        Turns         Reasoning
    ------------------------------------  ------------------------------------------------------  ----------------   ----------  ------------  ---------
    apollocorpus.parquet                  FreedomIntelligence/ApolloCorpus                        c1ee58d6           5,700,317   single-turn   no
    daring-anteater.parquet               nvidia/Daring-Anteater                                  ae79f8ac              99,532   multi-turn    no
    helpsteer2.parquet                    nvidia/HelpSteer2                                       990b2711              21,362   single-turn   no
    huatuogpt2.parquet                    FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K            4077ffae             142,248   single-turn   no
    medical-o1-reasoning-sft.parquet      FreedomIntelligence/medical-o1-reasoning-SFT            fc2c9e8a              90,120   single-turn   yes
    medical-o1-verifiable-problem.parquet FreedomIntelligence/medical-o1-verifiable-problem       46d5175e              40,644   single-turn   no
    medical-r1-distill-data.parquet       FreedomIntelligence/Medical-R1-Distill-Data             3491deec              22,000   single-turn   yes
    medmcqa.parquet                       openlifescienceai/medmcqa                               91c6572c             193,155   single-turn   no
    medquad.parquet                       lavita/MedQuAD                                          84ea67f8              16,407   single-turn   no
    medreason.parquet                     UCSC-VLAA/MedReason                                     7fc0ddb8              32,682   single-turn   yes
    miriad.parquet                        miriad/miriad-4.4M                                      ba958e43           4,487,542   single-turn   no
    ------------------------------------  ------------------------------------------------------  ----------------   ----------  ------------  ---------
    TOTAL                                                                                                           10,846,009

Full revision hashes:

    apollocorpus                   c1ee58d62a26a2422d01749517435fb0d716d449
    daring-anteater                ae79f8ac44cf185fbd3250dbe46057a7f7c4ec40
    helpsteer2                     990b2711a36180dd19d9c94b8627844866f8982a
    huatuogpt2                     4077ffaeb123e49b8b8a0283f42957a5570a52ce
    medical-o1-reasoning-sft       fc2c9e8a37b38f38da6d449564a8c350b244aef4
    medical-o1-verifiable-problem  46d5175eb74fdef3516d51d52e8c40db04bbdf35
    medical-r1-distill-data        3491deecebe1973a2d7370b824f4b41be29dcf1a
    medmcqa                        91c6572c454088bf71b679ad90aa8dffcd0d5868
    medquad                        84ea67f83cec9692ad254eaa02c9731b24ecfe4c
    medreason                      7fc0ddb80653c839be359ac4496ec6ee00bdd52f
    miriad                         ba958e430c08e3291aed5ffd3292f0962ade02e1


## Per-Dataset Details


### apollocorpus.parquet

    Source:    FreedomIntelligence/ApolloCorpus
    Revision:  c1ee58d62a26a2422d01749517435fb0d716d449
    Rows:      5,700,317
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: No
    Columns:   messages, source_dataset, subset

    Sample:
    {
      "messages": [
        {"role": "user", "content": "What are the advantages of using positron emission
         tomography (PET) to assess myocardial perfusion and metabolism compared to single
         photon emission computed tomography (SPECT)?"},
        {"role": "assistant", "content": "Positron emission tomography (PET) offers several
         advantages over single photon emission computed tomography (SPECT) when it comes to
         assessing myocardial perfusion and metabolism..."}
      ],
      "source_dataset": "FreedomIntelligence/ApolloCorpus",
      "subset": "eng_medqa"
    }


### daring-anteater.parquet

    Source:    nvidia/Daring-Anteater
    Revision:  ae79f8ac44cf185fbd3250dbe46057a7f7c4ec40
    Rows:      99,532
    Turns:     Multi-turn (up to 12 messages, some with system prompts)
    Reasoning: No
    Columns:   messages, source_dataset, dataset

    Sample:
    {
      "messages": [
        {"role": "user", "content": "Write a Python program that implements a handler that
         catches keyboard interrupts and saves the current state of the application before
         exiting"},
        {"role": "assistant", "content": "To create a Python program that implements a handler
         for keyboard interrupts (SIGINT)..."}
      ],
      "source_dataset": "nvidia/Daring-Anteater",
      "dataset": "code"
    }


### helpsteer2.parquet

    Source:    nvidia/HelpSteer2
    Revision:  990b2711a36180dd19d9c94b8627844866f8982a
    Rows:      21,362
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: No
    Columns:   messages, source_dataset, helpfulness, correctness, coherence, complexity, verbosity

    Sample:
    {
      "messages": [
        {"role": "user", "content": "c#"},
        {"role": "assistant", "content": "C# is a high-level, object-oriented programming
         language developed by Microsoft as part of its .NET initiative..."}
      ],
      "source_dataset": "nvidia/HelpSteer2",
      "helpfulness": 3,
      "correctness": 4,
      "coherence": 4,
      "complexity": 2,
      "verbosity": 1
    }


### huatuogpt2.parquet

    Source:    FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K
    Revision:  4077ffaeb123e49b8b8a0283f42957a5570a52ce
    Rows:      142,248
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: No
    Columns:   messages, source_dataset, id

    Sample:
    {
      "messages": [
        {"role": "user", "content": "我想咨询一下，在月经来的什么时候做爱最安全..."},
        {"role": "assistant", "content": "关于你的问题，我会分别进行解答..."}
      ],
      "source_dataset": "FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K",
      "id": "huatuogpt2_sft_1"
    }


### medical-o1-reasoning-sft.parquet

    Source:    FreedomIntelligence/medical-o1-reasoning-SFT
    Revision:  fc2c9e8a37b38f38da6d449564a8c350b244aef4
    Rows:      90,120
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: Yes (in "reasoning" column, chain-of-thought style)
    Columns:   messages, source_dataset, reasoning, subset
    Subsets:   EN, EN_mix, CN, CN_mix

    Sample:
    {
      "messages": [
        {"role": "user", "content": "Given the symptoms of sudden weakness in the left arm
         and leg, recent long-distance travel, and the presence of swollen and tender right
         lower leg, what specific cardiac abnormality is most likely to be found upon further
         evaluation that could explain these findings?"},
        {"role": "assistant", "content": "The specific cardiac abnormality most likely to be
         found in this scenario is a patent foramen ovale (PFO)..."}
      ],
      "source_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
      "reasoning": "Okay, let's see what's going on here. We've got sudden weakness in the
       person's left arm and leg...",
      "subset": "EN"
    }


### medical-o1-verifiable-problem.parquet

    Source:    FreedomIntelligence/medical-o1-verifiable-problem
    Revision:  46d5175eb74fdef3516d51d52e8c40db04bbdf35
    Rows:      40,644
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: No (short verifiable answers only)
    Columns:   messages, source_dataset

    Sample:
    {
      "messages": [
        {"role": "user", "content": "An 88-year-old woman with osteoarthritis is experiencing
         mild epigastric discomfort and has vomited material resembling coffee grounds multiple
         times. Considering her use of naproxen, what is the most likely cause of her
         gastrointestinal blood loss?"},
        {"role": "assistant", "content": "Gastric ulcer"}
      ],
      "source_dataset": "FreedomIntelligence/medical-o1-verifiable-problem"
    }


### medical-r1-distill-data.parquet

    Source:    FreedomIntelligence/Medical-R1-Distill-Data
    Revision:  3491deecebe1973a2d7370b824f4b41be29dcf1a
    Rows:      22,000
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: Yes (in "reasoning" column, chain-of-thought style)
    Columns:   messages, source_dataset, reasoning

    Sample:
    {
      "messages": [
        {"role": "user", "content": "A 30-year-old female presents with a history of itching
         under her right breast and has an annular ring lesion upon examination. What is the
         most likely fungal organism causing this condition?"},
        {"role": "assistant", "content": "The most likely fungal organism causing the annular
         ring lesion under the patient's breast is Trichophyton rubrum..."}
      ],
      "source_dataset": "FreedomIntelligence/Medical-R1-Distill-Data",
      "reasoning": "Okay, so I need to figure out the most likely fungal organism causing an
       annular ring lesion..."
    }


### medmcqa.parquet

    Source:    openlifescienceai/medmcqa
    Revision:  91c6572c454088bf71b679ad90aa8dffcd0d5868
    Rows:      193,155
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: No
    Columns:   messages, source_dataset, subject_name, topic_name, choice_type
    Note:      Multiple-choice QA; user message contains question + options (A/B/C/D)

    Sample:
    {
      "messages": [
        {"role": "user", "content": "Which of the following is derived from fibroblast
         cells ?\n\nA. TGF-13\nB. MMP2\nC. Collagen\nD. Angiopoietin"},
        {"role": "assistant", "content": "?. ?"}
      ],
      "source_dataset": "openlifescienceai/medmcqa",
      "subject_name": "Pathology",
      "topic_name": null,
      "choice_type": "single"
    }


### medquad.parquet

    Source:    lavita/MedQuAD
    Revision:  84ea67f83cec9692ad254eaa02c9731b24ecfe4c
    Rows:      16,407
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: No
    Columns:   messages, source_dataset, document_source, question_type, question_focus,
               umls_cui, umls_semantic_group
    Note:      Rich metadata including UMLS concept IDs and semantic groups

    Sample:
    {
      "messages": [
        {"role": "user", "content": "What is (are) keratoderma with woolly hair ?"},
        {"role": "assistant", "content": "Keratoderma with woolly hair is a group of related
         conditions that affect the skin and hair and in many cases increase the risk of
         potentially life-threatening heart problems..."}
      ],
      "source_dataset": "lavita/MedQuAD",
      "document_source": "GHR",
      "question_type": "information",
      "question_focus": "keratoderma with woolly hair",
      "umls_cui": "C0265964",
      "umls_semantic_group": "Disorders"
    }


### medreason.parquet

    Source:    UCSC-VLAA/MedReason
    Revision:  7fc0ddb80653c839be359ac4496ec6ee00bdd52f
    Rows:      32,682
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: Yes (in "reasoning" column, multi-path reasoning traces)
    Columns:   messages, source_dataset, reasoning, dataset_name, id_in_dataset

    Sample:
    {
      "messages": [
        {"role": "user", "content": "Urogenital Diaphragm is made up of the following,
         except:\n\nAnswer Choices:\nA. Deep transverse Perineus\nB. Perinial membrane\n
         C. Colle's fascia\nD. Sphincter Urethrae"},
        {"role": "assistant", "content": "Colle's fascia. Explanation: Colle's fascia does
         not contribute to the Urogenital Diaphragm..."}
      ],
      "source_dataset": "UCSC-VLAA/MedReason",
      "reasoning": "Finding reasoning paths:\n1. Urogenital diaphragm -> Pelvic anatomy ->
       Muscles and fascia...",
      "dataset_name": "medmcqa",
      "id_in_dataset": "abc123"
    }


### miriad.parquet

    Source:    miriad/miriad-4.4M
    Revision:  ba958e430c08e3291aed5ffd3292f0962ade02e1
    Rows:      4,487,542
    Turns:     Single-turn (1 user + 1 assistant)
    Reasoning: No
    Columns:   messages, source_dataset, paper_title, passage_text, year, venue, specialty
    Note:      Medical QA generated from scientific papers; includes source paper metadata

    Sample:
    {
      "messages": [
        {"role": "user", "content": "What are the common clinical features and diagnostic
         criteria of relapsing polychondritis?"},
        {"role": "assistant", "content": "The common clinical features of relapsing
         polychondritis include auricular chondritis, arthropathy, nasal chondritis,
         chondritis of the respiratory system..."}
      ],
      "source_dataset": "miriad/miriad-4.4M",
      "paper_title": "Otological Manifestations of Relapsing Polychondritis",
      "passage_text": "Relapsing Polychondritis is a rare systemic inflammatory disorder...",
      "year": 2015,
      "venue": "Indian Journal of Otolaryngology and Head & Neck Surgery",
      "specialty": "Otolaryngology"
    }
