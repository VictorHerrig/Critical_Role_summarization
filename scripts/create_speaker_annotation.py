import json
import re

from argparse import ArgumentParser

import numpy as np
import pandas as pd


SPEAKER_NAMES = [
    'MATT',
    'SAM',
    'LIAM',
    'LAURA',
    'ASHLEY',
    'TRAVIS',
    'TALIESIN',
    'MARISHA'
]


def main(
        alignment_json: str,
        output_path: str
):
    with open(alignment_json, 'r') as f:
        alignment_dict = json.load(f)

    current_speaker = 'UNKNOWN'
    speaker_annotations = list()

    # Iterate over text and extract the relevant speaker annotations
    # Could be sped up by putting in a function and doing a list comprehension, but it's a quick job anyway
    for filename, text in alignment_dict.items():
        # Anomalies
        in_parentheses = text[0] == '(' and text[-1] == ')'
        in_brackets = text[0] == '[' and text[-1] == ']'
        speaker_substring = re.search(r'([A-Z]+):', text)
        unknown_speaker = speaker_substring is not None and speaker_substring[1] not in SPEAKER_NAMES

        # Catch cases like '(all laugh)', '[dramatic music]' or unknown speakers (guests, etc.)
        if in_parentheses or in_brackets or unknown_speaker:
            current_speaker = 'UNKNOWN'
            speaker_annotations.append([filename, 'UNKNOWN'])
            continue

        # Find the speakers present in the caption
        present_speakers = [name in text for name in SPEAKER_NAMES]

        # Continue the current speaker if there is none present
        if not any(present_speakers):
            speaker_annotations.append([filename, current_speaker])
        # Switch speakers if there is a change at the beginning, else change the current speaker without annotating
        elif sum(present_speakers) == 1:
            speaker = SPEAKER_NAMES[present_speakers.index(True)]
            if text.index(speaker) <= 2:
                speaker_annotations.append([filename, speaker])
            # If it changed later on, label this as multiple and change the current speaker
            else:
                speaker_annotations.append([filename, 'MULTIPLE'])
            current_speaker = speaker
        # If there are multiple known speaker labels, label as multiple
        else:
            speaker_idxs = [-1 if not present else text.index(SPEAKER_NAMES[i])
                            for i, present in enumerate(present_speakers)]
            last_speaker = SPEAKER_NAMES[np.argmax(speaker_idxs)]
            current_speaker = last_speaker
            speaker_annotations.append([filename, 'MULTIPLE'])

    output_df = pd.DataFrame(data=speaker_annotations, columns=['filename', 'speaker'])
    output_df.to_csv(output_path, index=False, header=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--alignment-json', type=str, required=True, help='Path to the alignment json')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output .csv file')
    args = parser.parse_args()
    main(**vars(args))
