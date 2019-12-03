
% clear; load CLEF63Train; [X, Y] = creatSubTable(data_array, tree); save CLEF63TrainSubTable X Y tree;
% load VOC20Test; [Xt, Yt] = creatSubTable(data_array, tree); save VOC20TestSubTable Xt Yt tree;
clear;clc;close all;
%% ���ݼ�ѡ��
str1={'Cifar4096d'};% 'DD27';'Protein194';'VOC20';'Car196';'ILSVRC57'; 'AWAphog10';'CLEF63';'Cifar4096d';'Sun324'};%'Cifar4096d';'Sun324'�ڴ治��
m = length(str1);

%% ѵ�����̵Ĳ���
% para.alpha                % ||W||_21
% para.beta                 % ||W-mean(sum(W))||_F^2
para.p = 2;               % 1 or 2, Capped Lp-norm
para.NITER = 40;          % iter times��ǰ4������ȷ��epsilon
para.flag = 0;            % �Ƿ������������ 
%% ���Թ��̵Ĳ���
svm_opt = '-s 0 -c 1 -t 0 -q';
numFolds = 10;
%% ��ͬalpha��beta�µ�ѵ���Ͳ���
for iDataset = 1:m
    k = 1;
    DataName = str1{iDataset};
    load ([DataName 'TrainSubTable']);
    for iAlp = -3:5
        para.alpha = 10^iAlp;
        for iBeta = -3:5
            para.beta = 10^iBeta;
           %% Feature selection
            tic;
            [W, idx, OBJ, epsilon] = fun_Hier_CapLpSVM_W21_Wch(X, Y, tree, para);
            t_TrainFeature = toc;
            save([str1{iDataset} 'Trainfeature_HFFSR_' num2str(iAlp) '_' num2str(iBeta)],'W','idx','OBJ','epsilon','para');
            clear W epsilon OBJ;
           %% Classification for feature test
            load([DataName 'Test.mat']);
            if strcmp(DataName,'DD27') | strcmp(DataName,'Protein194')
                nFeaSel = round((size(data_array,2)-1)*0.1);
            else
                nFeaSel = round((size(data_array,2)-1)*0.2);
            end
            tic;
            [Model_SVM,TimeTest,RealLabel,OutLabel,Acc,AccStd,F_LCA,FH,TIE] = FeaSelc_K_Fold_SVM_hierarchical_classifier_lxx(data_array, tree, idx, nFeaSel, numFolds, svm_opt);
            t_TestFeature = toc;
           %% �������ı���
            Results(k,1:3) = [para.p, para.alpha, para.beta];
            Results(k,4) = nFeaSel;
            Results(k,5:11) =[Acc,AccStd,F_LCA,FH,TIE,t_TestFeature,t_TrainFeature];
            clear data_array;
            k = k+1;
        end
    end
    save([DataName '_HFFSR_SVM_result'], 'Results','idx');
end
disp('..... HierCapLpSVM21Wch ... Finshed! ....')

%% ��ÿһ��ķ�����ȷ��
% nClass = 27;
% AccPerClass = zeros(nClass,11);
% for icell = 1:10
%     for iClass = 1:nClass
%         tempIdex = find(RealLabel{icell} == iClass);
%         nSample = length(tempIdex);
%         tempReal = RealLabel{icell}(tempIdex);
%         tempOut = OutLabel{icell}(tempIdex);
%         nRight = length(find(tempReal == tempOut));
%         if nSample~=0
%             AccPerClass(iClass,icell) = double(nRight/nSample);
%         end
%     end
% end
% for iClass = 1:nClass
%     AccPerClass(iClass,11) = double(mean(AccPerClass(iClass,1:end-1)));
% end

%% ͼ����ʾ����Ҷ�ӽ���W
% load('D:\setup\MatlabR2016a\bin\lxx\SVM_Capped\Results\DD27Trainfeature_HFFSR_1_1.mat')
% iNode = 32;
% dFea  = 473;
% MAX = repmat(max(W{iNode}),[dFea,1]);
% MIN = repmat(min(W{iNode}),[dFea,1]);
% WW = (W{iNode} - MIN)./(MAX - MIN + eps);
% imagesc(WW)
% 

%% ����i�����j����ͬʱѡ��������������ص��ȡ����ض�
% FeaIdx = zeros(200,10); 
% % ��ճ����ȥ��ѡ����������
% length(setdiff(FeaIdx(:,9), FeaIdx(:,6)))         % �������������
% length(intersect(FeaIdx(:,9), FeaIdx(:,6)))       % ��������������������
% % Ȩ���ص��ȡ����س̶ȣ�200��������Ȩֵ��һ��[0,1],����������Ȩֵ֮�͡�����������Ȩֵ֮�� 
% Wweight = [sum(W{21}.^2, 2),sum(W{22}.^2, 2),sum(W{23}.^2, 2), sum(W{24}.^2, 2),sum(W{25}.^2, 2),...
%     sum(W{26}.^2, 2),sum(W{27}.^2, 2),sum(W{28}.^2, 2),sum(W{29}.^2, 2),sum(W{30}.^2, 2)];
% Wweight2 = Wweight;
% for i = 1:10
%     Wweight2(FeaIdx(:,i),i) = mapminmax(Wweight(FeaIdx(:,i),i)',0,1)';  % ÿһ����£���ѡ����Ȩ�ع�һ����
% end
% FeaCom = intersect(FeaIdx(:,9), FeaIdx(:,2));   % ǰ������߹��������
% SimD = sum(Wweight2(FeaCom,1))%/length(FeaCom)
% 
% FeaDif = setdiff(FeaIdx(:,9), FeaIdx(:,2));     % ǰ�߶��е�����
% DifD = sum(Wweight2(FeaDif,1))%/length(FeaDif)

%% iNode��ѡ����������jNode�Ϸ���
% iNode = 29; jNode = 22;     % iNode��������jNode�Ϸ���
% nFeaSel = 200;
% Fea_Sel_id = idx{iNode}(1:nFeaSel);
% 
% Kfold = 10;
% svm_opt =  '-s 0 -c 1 -t 0 -q';
% 
% Acc = zeros(1,Kfold);
% 
% load VOC20TestSubTable;
% data_array = [Xt{jNode},Yt{jNode}];     % jNode�������ӱ�
% [M,~]=size(data_array); 
% 
% rand('seed',1);
% indices = crossvalind('Kfold',M,Kfold);
% for k = 1:Kfold
% 	testID = (indices == k);    % test��Ԫ�������ݼ��ж�Ӧ�ĵ�Ԫ���
% 	trainID = ~testID;          % train��Ԫ�صı��Ϊ��testԪ�صı��        
% 	train_array = data_array(trainID,:);
% 	test_array = data_array(testID,:);
%             
% 	x = train_array(:,Fea_Sel_id);
% 	y = train_array(:,end);
% 	ModelSVM = svmtrain(y,x,svm_opt);
%         
% 	test_data = test_array(:,Fea_Sel_id);
% 	test_label = test_array(:,end);       
% 	[~,acc, ~] = svmpredict(test_label,test_data,ModelSVM,'-q');
%     Acc(1,k) = acc(1);
% end
% AccMean = mean(Acc)  
% AccStd = std(Acc)