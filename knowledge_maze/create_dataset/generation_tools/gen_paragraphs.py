async def create_paragraphs(qa_data: list[dict]) -> dict:
    """Create the paragraph dataset"""
    # Collect the id & question pairs lists:
    middle_id_list = []
    middle_pargraph_info_list = []
    edge_id_list = []
    edge_pargraph_info_list = []

    for sample in qa_data:
        middle_id_list += get_middle_paragraph_ids(sample)
        middle_pargraph_info_list += get_middle_paragraph_info(sample)
        edge_id_list += get_edge_paragraph_ids(sample)
        edge_pargraph_info_list += get_edge_paragraph_info(sample)

    # Create all the paragraphs:
    middle_paragraphs = await generate_paragraph(middle_pargraph_info_list, mode='Chain')
    edge_paragraphs = await generate_paragraph(edge_pargraph_info_list, mode='Edge')

    # Pair ids with paragraphs:
    paragraph_data = {}
    for index, paragraph in zip(middle_id_list, middle_paragraphs):
        paragraph_data[index] = paragraph
    for index, paragraph in zip(edge_id_list, edge_paragraphs):
        paragraph_data[index] = paragraph

    return paragraph_data


def get_middle_paragraph_ids(qa_sample: dict) -> list[str]:
    path_1_ids = [sample['Paragraph'] for sample in qa_sample['Reasoning path 1'] if sample['Next question']]
    path_2_ids = [sample['Paragraph'] for sample in qa_sample['Reasoning path 2'] if sample['Next question']]
    return path_1_ids + path_2_ids


def get_edge_paragraph_ids(qa_sample: dict) -> list[str]:
    path_1_ids = [sample['Paragraph'] for sample in qa_sample['Reasoning path 1'] if not sample['Next question']]
    path_2_ids = [sample['Paragraph'] for sample in qa_sample['Reasoning path 2'] if not sample['Next question']]
    return path_1_ids + path_2_ids


def get_middle_paragraph_info(qa_sample: dict) -> list[str]:
    path_1_info = [
        f"""Question 1: {sample['Previous question']} \n
            Question 2: {sample['Next question']} \n
            Answer": {qa_sample['Answer']['Memory answer']}"""
        for sample in qa_sample['Reasoning path 1'] if sample['Next question']
    ]
    path_2_info = [
        f"""Question 1: {sample['Previous question']} \n
            Question 2: {sample['Next question']} \n
            Answer": {qa_sample['Answer']['Memory answer']}"""
        for sample in qa_sample['Reasoning path 2']  if sample['Next question']
    ]
    return path_1_info + path_2_info


def get_edge_paragraph_info(qa_sample: dict) -> list[str]:
    path_1_info = [
        f"""Question: {sample['Previous question']} \n
            Answer": {qa_sample['Answer']['Counter answer 1']}"""
        for sample in qa_sample['Reasoning path 1'] if not sample['Next question']
    ]
    path_2_info = [
        f"""Question: {sample['Previous question']} \n
            Answer": {qa_sample['Answer']['Counter answer 2']}"""
        for sample in qa_sample['Reasoning path 2']  if not sample['Next question']
    ]
    return path_1_info + path_2_info