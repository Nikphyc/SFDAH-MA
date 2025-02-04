function [] = demo_eval_PWCF(n_bits, domain, dataset, prefix)
     addpath('utils/');

%    n_bits  = 64;
%    dataset = 'Office-Home';
%    prefix  = '0526';

    % load mat
    fprintf('Load hashcodes and labels of %s.\n', dataset);
    hashcode_path = sprintf('../hashcode/HASH_%s_%s_%dbits.mat', dataset,domain,n_bits);
    fprintf(hashcode_path);
    load (hashcode_path);
    fprintf('End Load hashcodes and labels of %s.\n', dataset);

    % get label
    trn_label = double(L_db); % 2427*1
    tst_label = double(L_te); % 436*1
    fprintf('trn_label 的长度为：%d\n', length(trn_label));%2427
    fprintf('tst_label 的长度为：%d\n', length(tst_label));%436




    query_retrieval = repmat(tst_label,1,length(trn_label)); % t*s
    retrieval_query = repmat(trn_label,1,length(tst_label)); % s*t
    % 计算 cateRetriTest
    cateRetriTest = false(numel(tst_label), numel(trn_label)); % 初始化结果矩阵

    for i = 1:numel(tst_label)
        for j = 1:numel(trn_label)
            cateRetriTest(i,j) = tst_label(i) == trn_label(j);
        end
    end
    %cateRetriTest   = (query_retrieval==retrieval_query'); % t*s

    
    % get hash code
    tB = compactbit(val_B > 0);
    B = compactbit(retrieval_B > 0);
    hamm = hammingDist(tB, B); % t*s

    % eval
    [recall, precision, ~] = recall_precision(cateRetriTest, hamm);
    [mAP] = area_RP(recall, precision);
    
    clear hamm;
    clear cateRetriTest;


    % the final results
    fprintf('--------------------Evaluation: mAP@PWCF-------------------\n')
    fprintf('mAP@PWCF = %04f\n', mAP);

    % save
    % result_name = ['../results/' prefix '_' dataset '_' num2str(n_bits) 'bits_' datestr(now,30) '.mat'];
    result_name = ['../results/' prefix '_' dataset '_' domain '_' num2str(n_bits) 'bits' '.mat'];
    save(result_name, 'precision', 'recall', 'mAP');







