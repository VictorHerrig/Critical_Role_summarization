# CRD3_summarization
## Bidet!
### Critical Role episode summarization.

## Explanation

Dungeons&Dragons is a tabletop role-playing game in which player assume the
role of, usually, one character (PC). PCs have social interactions,
battle, and explore the world as presented by the dungeon master (DM). The
DM plays all non-PC characters (NPCs) and acts as a narrator by describing
all aspects of the world and the inner minds of PCs and NPCs.

Critical Role is a live-streaming show on Twitch in which a core cast of
players, the DM and occasional guests play long-running and shorter-form
Dungeons&Dragons games. It has become very popular (boasting the highest
revenue of all Twitch streams) and the fan base has made available
transcripts and summaries of episodes.

## The General Idea
This project has grown in my mind from being a playground for summarization to perhaps a truly useful application. The idea is this:
 - Listen to an incoming twitch stream using rtmpdump or similar, getting only the audio (although video may theoretically provide further accuracy, I suspect there will be serious diminishing returns)
 - Have a constantly running speaker-attributed automatic speech recognition (SA-ASR) system running that produces a 'script' of sorts; this may be feasible on CPU, but we'll have to see
 - At fixed intervals (say, 1 or 5 minutes) run a series of summarization inference tasks on GPU to produce summaries for a set of durations (e.g. the last 1 minute, 5 mintues, 15 mintues, 1 hour...)
 - These different time periods could, naively, be simply pasted together from finer-grained summaries, though a better approach would likely be to do some prompt engineering using previous output summaries
So, this will provide listeners the ability to step away to do whatever they need without fear of missing out of story details. Twitch chat is notoriously unhelpful when queried, so this may serve as a sort of ad-hoc replacement to human summarization.
Naturally, if this were to be made into an actual running service, there would be plenty of ops questions to sort out, but for the time being I am focusing on development of a baseline system.

## Requirements
Aside from those packages listed in setup.py, you will need to additionally install:
- cuda-toolkit (this can be done system-wide or with conda using the nvidia channel)
- cuda-nvcc (this can be done system-wide or with conda using the nvidia channel)
- LLVM (this can be done system-wide or with conda using the nvidia channel)
- TVM (you will need to build from source using the following options in config.cmake: 
  USE_CUDA YES; (USE_LLVM "/path/to/llvm-config --link-static"); set(HIDE_PRIVATE_SYMBOLS ON)).
  This may frankly be rather trying
- Any C++ compilation tools required to build TVM on your system
- 

## Acknowledgements

CRD3 Data is produced under the Creative Commons Attribution-ShareAlike 4.0
International License. The repository can be found at
https://github.com/RevanthRameshkumar/CRD3, or through the submodule link in
this repository. The original paper detailing the creation of the dataset can
be found on ACL Anthology: https://aclanthology.org/2020.acl-main.459.pdf.

The transcript from campaign 3 I use for testing, as well as 'episode blurbs'
are collected from the Critical Role fandom:
https://criticalrole.fandom.com/wiki/List_of_episodes.

Diagonaled matrix multiplication CUDA kernel code is adapted from the
Longformer repository (under the Apache 2.0 License) developed by the
Allen Institute for AI, found here: https://github.com/allenai/longformer.

The approach used in this repository is adapted from the top-down,
bottom-up, long-range transformer approach developed by Salesforce Research.
There is no public repository for this work, though the paper is available
on Arxiv: https://arxiv.org/pdf/2203.07586v1.pdf.

## Let chaos reign!
