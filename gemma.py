
# Path
if True:
    path_model = r'/data/@Zilong_Works/Data/Experiment/SLAKE/Gemma/Gemma3_fine/gemma-3-4b-it'
    root_save_image = r'/data/@Zilong_Works/Data/SourceData/Dataset/VQA/SLAKE/datasets--BoKelvin--SLAKE/imgs/imgs/'
    path_dataset = r'/data/@Zilong_Works/Data/SourceData/Dataset/VQA/SLAKE/datasets--BoKelvin--SLAKE'
    root_result = r'/data/@Zilong_Works/Data/Experiment/SLAKE/Gemma/Gemma3/'

# import
if True:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from datasets import load_dataset
    from PIL import Image
    from datasets import load_dataset
    from transformers import AutoProcessor
    import torch
    import pickle
    import os

# format
if True:
    def format_data(example):
        system_message = "You are a helpful assistant."
        question = example["question"]
        answer = example["answer"]
        img_name = example["img_name"]
        img_path = root_save_image+img_name
        img = Image.open(img_path).convert("RGB")
        # print(img)  # todo
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                        {
                            "type": "image",
                            "image": img,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ],
        }

    def process_vision_info(messages: list[dict]) -> list[Image.Image]:
        image_inputs = []
        # Iterate through each conversation
        for msg in messages:
            # Get content (ensure it's a list)
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            # Check each content element for images
            for element in content:
                if isinstance(element, dict) and (
                    "image" in element and element.get("type") == "image"
                ):
                    # Get the image and convert to RGB
                    if "image" in element:
                        image = element["image"]
                    else:
                        image = element
                    # print(type(image))  # todo
                    image_inputs.append(image.convert("RGB"))
        return image_inputs

# test init
if True:
    # Load Model with PEFT adapter
    model = AutoModelForImageTextToText.from_pretrained(
        path_model, device_map="balanced",)
    processor = AutoProcessor.from_pretrained(path_model)

# test
if True:
    dataset_test = load_dataset(path_dataset, split="test")
    
    dataset_test=dataset_test.select(range(30))
    
    data_test = [format_data(sample) for sample in dataset_test]
    path_test_checkpoint = root_result + 'test_checkpoint_OriginalGemma.pkl'

    # 判断是否已经有test_checkpoint.pkl文件，有则读取，没有则创建
    if True:
        if os.path.exists(path_test_checkpoint):
            with open(path_test_checkpoint, 'rb') as f:
                out_list = pickle.load(f)
                start_i = len(out_list)
        else:
            out_list = []
            start_i = 0

    for i in range(start_i, len(data_test)):
        case = data_test[i]
        messages = case["messages"]
        message_input=messages[0:2]
        print('message_input:', message_input)
        message_answer=messages[2]

        inputs = processor.apply_chat_template(
            message_input, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        
        question = messages[1]["content"][0]["text"]
        answer = messages[2]["content"][0]["text"]
        print('==='+str(i)+'/'+str(len(data_test))+'===')
        print('question:', question)
        print('answer:', answer)
        print('decoded:', decoded)

        save_one_sample = {
            'one_case': {
                'question': question,
                'answer': answer,
                'image': messages[1]["content"][1]["image"],
            },
            'pred_answer': decoded,
        }
        out_list.append(save_one_sample)

        # 每50个保存一次
        if (i+1) % 50 == 0:
            with open(path_test_checkpoint, 'wb') as f:
                pickle.dump(out_list, f)
            print('saved:', str(i+1), 'out_list:', out_list)

    # 最后保存一次
    with open(path_test_checkpoint, 'wb') as f:
        pickle.dump(out_list, f)
    print('out_list:', out_list)


pass
