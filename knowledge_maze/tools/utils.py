import textwrap


def print_question_sample(sample, width=120):
    """
    Print a question sample in a readable format with clear sections.

    Args:
        sample (dict or list): A dictionary containing question data or a list with the sample at index 0
        width (int): Maximum width for text wrapping
    """
    # Handle the case where sample is a list
    if isinstance(sample, list):
        sample = sample[0]

    print(f"\n{'=' * width}")

    # Basic question and answer info
    print(f"\nQUESTION: {sample['question']}")

    # Answer information with highlighting
    print("\nANSWER DETAILS:")
    print(f"  Memory based response:     {sample['answer']['memory']}")
    print(f"  Claim based response:      {sample['answer']['claim']}")
    print(f"  Evidence based response:   {sample['answer']['evidence']}")
    print(f"\nREASONING TYPE: {sample['reasoning_type']}")

    print(f"\n{'=' * width}")

    # Context information
    print("\nCONTEXT:")
    for context_type, content in sample['context'].items():
        print(f"\n  {context_type.upper()}:")

        # Handle both string and list contexts
        if isinstance(content, list):
            for i, paragraph in enumerate(content):
                print(f"    Paragraph {i + 1}:")
                # Wrap the text and indent each line for readability
                wrapped_text = textwrap.fill(
                    paragraph,
                    width=width - 8,  # Account for indentation
                    initial_indent="      ",
                    subsequent_indent="      "
                )
                print(wrapped_text)
                print()  # Add a blank line between paragraphs
        else:
            # Wrap the text and indent each line for readability
            wrapped_text = textwrap.fill(
                content,
                width=width - 8,  # Account for indentation
                initial_indent="      ",
                subsequent_indent="      "
            )
            print(wrapped_text)

    print(f"{'=' * width}")


def print_question_samples(samples, width=120):
    """
    Print multiple question samples in a readable format.

    Args:
        samples (list): A list of dictionaries containing question data
        width (int): Maximum width for text wrapping
    """
    if samples and not isinstance(samples[0], dict):
        # If the first item is not a dict, assume we have a nested structure
        samples = samples[0]

    for i, sample in enumerate(samples):
        print_question_sample(sample, i)

    print(f"\n{'=' * width}")
    print(f"SUMMARY: Displayed {len(samples)} question samples")
    print(f"{'=' * width}\n")