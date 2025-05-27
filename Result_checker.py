
if True:
    import pickle
    path_pkl_test = r'/data/@Zilong_Works/Data/Experiment/SLAKE/Gemma/Gemma3/test_checkpoint_OriginalGemma.pkl'
    test_record_list = pickle.load(open(path_pkl_test, 'rb'))

    # 对每一个样本进行check，假如在pred中找得到answer中的每个词，则视为正确
    len_result = len(test_record_list)
    check_result_list = []
    for i in range(len_result):        
        one_result = test_record_list[i]
        answer = one_result['one_case']['answer']
        pred_answer = one_result['pred_answer']
        
        true_answer_list= answer.split(' ')
        match_list = []
        for true_answer in true_answer_list:
            if true_answer in pred_answer:
                match_list.append(true_answer)
        match_rate = len(match_list) / len(true_answer_list)
        check_result_list.append(match_rate)
    # 计算平均值
    avg_match_rate = sum(check_result_list) / len(check_result_list)
    print('avg_match_rate:', avg_match_rate)
    
pass 
