import json
import re

from argparse import ArgumentParser
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

    current_speaker = ''
    speaker_annotations = list()

    # Iterate over text and extract the relevant speaker annotations
    # Could be sped up by putting in a function and doing a list comprehension, but it's a quick job anyway
    for filename, text in alignment_dict.items():
        # Anomalies
        in_parentheses = text[0] == '(' and text[-1] == ')'
        in_brackets = text[0] == '[' and text[-1] == ']'
        speaker_substring = re.search(r'([A-Z]+):', text)
        unknown_speaker = speaker_substring is not None and speaker_substring[1] not in SPEAKER_NAMES

        # Catch cases like '(all laugh)', '[dramatic music]' without forgetting the current speaker
        if in_parentheses or in_brackets:
            continue
        # Catch untracked speakers (guests, etc.) and reset the current speaker
        if unknown_speaker:
            current_speaker = ''
            continue

        # Find the speakers present in the caption
        present_speakers = [name in text for name in SPEAKER_NAMES]

        # Continue the current speaker if there is none present
        if not any(present_speakers) and current_speaker in SPEAKER_NAMES:
            speaker_annotations.append([filename, current_speaker])
        # Switch speakers if there is a change at the beginning, else change the current speaker without annotating
        if sum(present_speakers) == 1:
            speaker = SPEAKER_NAMES[present_speakers.index(True)]
            if text.index(speaker) <= 2:
                speaker_annotations.append([filename, speaker])
            current_speaker = speaker

        # Continue if there are multiple speakers or if there are no speakers and also no current speaker

    output_df = pd.DataFrame(data=speaker_annotations, columns=['filename', 'speaker'])
    output_df.to_csv(output_path, index=False, header=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--alignment-json', type=str, required=True, help='Path to the alignment json')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output .csv file')
    args = parser.parse_args()
    main(**vars(args))
