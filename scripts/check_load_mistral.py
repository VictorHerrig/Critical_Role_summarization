# This script simply tests that you can load the model into GPU memory
import torch
from transformers import GenerationConfig

from unsloth import FastLanguageModel


def main():
    model, tokenizer = FastLanguageModel.from_pretrained('unsloth/mistral-7b-bnb-4bit', load_in_4bit=True, device_map='auto')
    model.load_adapter('unsloth_mistral_train/checkpoint-256')

    print('\n------------')
    print('Model params')
    print('------------\n\n')
    for param_name, param in model.named_parameters():
        print(f'{param_name}: {tuple(param.shape)}')
    print(model.device)
    print(f'\n\nTokenizer vocab size: {tokenizer.vocab_size}')

    prompt_str = """<s> [INST] The following is an excerpt from a show in which multiple actors are playing Dungeons and Dragons:

 LAURA: It wasn't Power Word Kill, it was Finger ofDeath?
 MATT: No, he'd already used his Power Word Kill. Remember, he's not a god yet."
 ASHLEY: Okay.
 LAURA:Just the one. Just the one Power Word Kill.
 TRAVIS: The fact that he's not a god gives me diarrhea feelings.
 ASHLEY: Okay.
 MATT: All right, and he's going to move back a bit, frustrated at the--
 SAM: At everything.
 MATT: At your attempts to Counterspell. Actually, no. He laughed at that. "Nice try."
 SAM: It wasn't my best performance.
 MATT: All right, that's going to end his turn. Delilah's out. Grog.
 TRAVIS: (laughs)
 MATT: Make a stealth check.
 TRAVIS: Cool. Right. Pass Without a Trace is still going, right?
 MATT: Nope.
 TRAVIS: That's a one.
 TALIESIN: Hello?
 MATT : As you're there, Grog, you hear (screeching). A cloud of screeches, as about nine of these gloom stalkers all start (whoosh) going into a dive towards you.
 SAM: Maybe one of them is our friend Mr.Mistoffelees.
 MARISHA: Mr. Mistoffelees, come back!
 TRAVIS: Am I going to do anything? How far away is the nearest structure?
 MATT:Nothing. It's a crater. You're where the city used to be. You have at least a mile to each side of nothing but crater.
 TALIESIN: Fuck!
 ASHLEY: He's all by his lonesome!
 TALIESIN: One more round.
 SAM: He doesn't know.
 MARISHA: He could be stuck there forever!
 TALIESIN: There's nothing we can do right now.
 TRAVIS: I'm going to go into a frenzied rage, and I'm going to use my second wind to heal 1d10 plus my fighter level. Three, six. Six points, yay.
 TALIESIN: How close to you do we have to be?
 MARISHA: We have to be touching.
 SAM: And you have to not be wind.
 MARISHA: I can be wind now.
 TRAVIS: I don't have any fighter things, so that's my turn.
 MATT: That's your turn. Okay. Ending your turn. Dark energy suffuses the landscape in the area. Actually, no. I'm not going to use that one because nobody has necrotic resistance right now. More undead are going to pop up. Let's see how many. That's actually a shitty roll. Four undead pop up.
 TALIESIN: They're bullshit. We don't care."
 MATT: All righty. That's going to be an attack on you. It has advantage because you're prone and it's in melee. That's a 17. Still doesn't hit. High armor. You're ducking out of the way. The skeleton is (tink) hitting the ground. It's also attacking at you, Pike. That's cocked. Natural one. It goes to strike you, and it doesn't even hit you. The holy symbol of Sarenrae flares, and it (raspy cry) pulls away from the blast of vibrant, radiant energy. There's going to be one attacking you, Scanlan. That is a 20 to hit?
 
That is the end of the script.
Write a summary of the script, making sure to include all the major events contained within.
 [/INST]
    """

    # prompt_str = """<s> [INST] The following is an excerpt from a show in which multiple actors are playing Dungeons and Dragons:
#
#  MATT: 24. So in setting up preparation, you take each one of the heads, and to fill each one of their-- I want to say concave, but it's an interior pyramid space. Convex is outward. It will be about 100 platinum pieces or so to fill each face void. So you take the platinum, you melt it down into the smelter, get it to where it's in the gripped, tong-held metallic reservoir, until it eventually melts down. You then pour it into each of the heads, so 300 platinum utilized to do this. It fills to the very edge. Some of it spills over, but you can easily snip that off. As you wait for it to cool, which you can probably help with, actually, as there is no basin for quenching the metal at the moment, so the steam (whoosh) rises up all around, and you turn the heads over, and each one (sliding sound) leaves these heavy metallic pyramid pieces that are smooth and perfect on three of the sides, and one of them appears to be almost like a broken crystal on the inside. It's a shattered, messed-up section.
#  LAURA: Puzzle piece! Got to fit them together!
#  TALIESIN: Okay, so if we fit these together--
#  MATT: They seem to fit together very well; you figured it out very quickly, they fit and hold together.
#  LAURA: With the jaggedy ends, or with the smooth ends?
#  MATT: The jaggedy ends all fit together.
#  TALIESIN: Does this now make another concave piece?
#  MATT: All together, it makes a pyramid that comes to a central point.
#
# That is the end of the script.
# Write a summary of the script, making sure to include all the major events contained within.
#  [/INST]
# """

# prompt_str = """<s> [INST] The following is an excerpt from a show in which multiple actors are playing Dungeons and Dragons:
#
#  MATT: As you guys walk forward towards the singular hallway, you go through the domed arch, and the hallway extends on. Featureless, smooth grey marble. Like a solid punch that seemed to have carved out whatever this material was. It extends for 100 feet. 150 feet. And even though there's very very faint light, and many of you can see through darkvision, there is a darkness about 60 feet from you that seems to consume what you can see. You can only see so far beyond that radius, and you're not quite sure where that oppressive shadow is coming from. The air is chilled, but not uncomfortably so. You can hear the distant sound of pinched winds whining through small spaces. A familiar sound from the many times you've spent underground, through caverns that had some openings that allowed air to streak through to the surface. A few steps more and suddenly the hallway opens into a second chamber. Into a library, if you could even call it that. Bookshelves that stretch upward for seemingly hundreds of feet, before flickering lights fell from your vision. You do not see where they stop. These columns of just tomes and cases and scrolls and parchments stacking as in tablets and any form of recorded information seem to just be infinite in front of you, to two sides of this road and upward. They seem to stretch in almost every direction, the walls to the side of you. You see columns ahead of you, and they tend to wind and curl like a labyrinth made of bookcases, to no end. It's very Escher-esque in how the further up you look, they tend to curve and shift and bend around each other. Your eyes, now attuned to the realms beyond mortality, you catch glimpses of shifting wind gathering books and turning pages. Spirits of the library, though sparse as they may be from this entry point, organizing. Reading and learning from whatever words are within. You begin to almost get vertigo, trying to contemplate the strange and overwhelming construction of such a wondrous place. An endless sea of stories. Spectral custodians keeping the tales of each other and the history of Exandria past, present, and binding the pages to tell what will eventually be the present, currently the future.
#
# That is the end of the script.
# Write a summary of the script, making sure to include all the major events contained within.
#  [/INST]
# """

    print('\n-----------')
    print('   Prompt')
    print('-----------\n\n')
    print(prompt_str)
    print('\n\n')
    
    # ground_truth = "They step through a hallway into an immense library, tended by spectral custodians. Percy eagerly starts examining books and quickly realizes that each tome is a single person's life story, from birth to death, written in Celestial. Scanlan, clutching his Ioun Stone, starts walking purposefully toward what they hope is the center of the library. As Percy and Vex fall in behind him, Scanlan starts playing a wedding march on his flute, a nod to some information that Percy had let slip during Pelor's trial."

    ground_truth = "At the top, they engage in combat with the three figures spotted earlier: a death knight, Delilah Briarwood, and the newly reborn Undying King, Vecna. The fight goes badly; several of them are immediately paralyzed; Vex is felled by Power Word Kill; Grog gets banished back to the Shadowfell; and Vax gets Disintegrated."
    
    print('Actual Summary')

    print('\n----------------')
    print(' Actual Summary')
    print('----------------\n\n')
    print(ground_truth)
    print('\n\n')

    prompt = torch.tensor(tokenizer.encode(prompt_str, add_special_tokens=False)).to(torch.int).reshape(1, -1)

    print(f'Prompt size: {prompt.size()}')

    generation_config = GenerationConfig(
        max_new_tokens=min(512, prompt.size()[1] // 3),
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    prompt = prompt.to(model.device)
    output = model.generate(
        prompt,
        generation_config=generation_config
    )

    print(f'Output size: {output.size()}')

    # For multiple output beam search
    # for i in range(output.sequences.size()[0]):
    #     print('\n-----------')
    #     print(f'Sequence {i}')
    #     print('-----------\n\n')
    #     print(f'Score for sequence {i}: {output.sequences_scores[i]}\n')
    #     print(f'Output:\n\n')
    #     print(dataset.construct_string(output.sequences[i, :].squeeze()))
    #     print('\n\n')

    # For single output
    print('\n----------')
    print(f'Sequence')
    print('----------\n\n')
    print(f'Output:\n\n')
    print(tokenizer.decode(output.squeeze(), skip_special_tokens=False).split('[/INST]')[-1])
    print('\n\n')


if __name__ == '__main__':
    main()
