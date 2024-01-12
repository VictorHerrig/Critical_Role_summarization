# This script simply tests that you can load the model into GPU memory


from CRD3_summarization.models.HuggingfaceModels import QuantModelFactory


def main():
    model, tokenizer = QuantModelFactory.mistral_7b()
    for param_name, param in model.named_parameters():
        print(f'{param_name}: {tuple(param.shape)}')
    input()  # Wait and check VRAM


if __name__ == '__main__':
    main()
