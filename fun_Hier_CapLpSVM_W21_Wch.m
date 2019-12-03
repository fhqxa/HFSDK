%% fun_Hier_CapLpSVM_W21_Wch function
% min_{M>=0,W}  ||X'*W+1*b'-Y-B.*M||_cap2p + alpha*||W||_2,1 + beta *||W_s
% - sum(W_c)/c||_F^2
% Writted by Xinxin Liu, 2019-10-19

%% 变量注释
% Y must be 1 or -1
% X： d*n特征集；   
% Y： n*c标签
% r:  gama
% p: 0<p<=2
% W:  特征选择矩阵    b:偏置矩阵
% idx:特征选择后，各原始特征的顺序
% obj:优化目标值
% accuracy: 分类正确率
% idxe：
%%
function [W, idx, OBJ, epsilon] = fun_Hier_CapLpSVM_W21_Wch(X, Y, tree, para)
    %% parameter setting
    alpha = para.alpha;     % ||W||_21
    beta = para.beta;       % ||W-mean(sum(W))||_F^2
    p = para.p;             % 1 or 2, Capped Lp-norm
    NITER = para.NITER;     % iter times，the former four iteration is used to calculate epsilon
    flag = para.flag;
    clear para;
    %% basic information from tree hierarchy of classes
    nNode = size(tree,1);
    internalNodes = tree_InternalNodes(tree);
    indexRoot = tree_Root(tree);                % The root of the tree
    noLeafNode =[internalNodes;indexRoot];
    %% get the number of features
    [~,d] = size(X{indexRoot});
    %% get the internal nodes whose chilren are not internal nodes (interNodesNIN)
    indexLeaf = tree_LeafNode(tree);
    index = 1:nNode;
    index(indexLeaf) = [];
    treeTemp = tree(index,:);
    interNodesNIN = index(tree_LeafNode(treeTemp));  
    clear indexLeaf index treeTemp;
    %% compute C_max and get the number of no-internal nodes which are child node of current internal node
    m = zeros(1,nNode);
    nChildInterNode = zeros(1,nNode);
    ChildInterNode = cell(nNode,1);
    for i = 1:length(noLeafNode)
        iNode = noLeafNode(i);
        ClassLabel = unique(Y{iNode});
        m(iNode) = length(ClassLabel);
        ChildInterNode{iNode} = noLeafNode(tree(noLeafNode,1) == iNode);
        nChildInterNode(iNode) =length(ChildInterNode{iNode}) ;
    end
    Cmax = max(m);
    clear ClassLabel m;
    %% initialize
    nSam = zeros(nNode,1);
    nClass = zeros(nNode,1);
    for j = 1:length(noLeafNode)
        iNode = noLeafNode(j);
        X{iNode} = (X{iNode})';
        Y{iNode} = conversionY01_extend(Y{iNode},Cmax);   % Class label is extended, for example: 2 to [1 0]
        W{iNode} = rand(d, Cmax);                         % initialize W
        
        [~, nSam(iNode)] = size(X{iNode});         % d特征数目，n测试样本数
        nClass(iNode) = size(Y{iNode},2);                         % The number of classes
 
        M{iNode} = zeros(nSam(iNode),nClass(iNode));	% 松弛变量slack variable
        d1{iNode} = sparse(ones(nSam(iNode),1));      
        D1{iNode} = diag(d1{iNode});      % d1是n×1的列矩阵，元素都是1；D1为nSam×nSam的单位矩阵（初始化D）
        d2{iNode} = sparse(ones(d,1));         
        D2{iNode} = diag(d2{iNode});      % d2是d×1的列矩阵，元素都是1；D2是d×d的单位矩阵
        D3{iNode} = diag(0.5./max(sqrt(sum(W{iNode}.*W{iNode},2)),eps));
        H{iNode} = D1{iNode} ;%- 1/nSam(iNode) * ones(nSam(iNode));
    end
    %% iteration and updates
    OBJ = zeros(NITER,1);
    obj = cell(nNode,1);
    for iter = 1:NITER
        for j = 1:length(noLeafNode)
            iNode = noLeafNode(j);
            if(isempty(Y{iNode})==0) 
            Z{iNode} = Y{iNode} + Y{iNode} .* M{iNode};
            H{iNode} = D1{iNode} - 1/sum(d1{iNode})* (d1{iNode} * d1{iNode}');
            if ismember(iNode,interNodesNIN)
                W{iNode} = (X{iNode} * H{iNode} * X{iNode}' + alpha * D3{iNode} + beta * D2{iNode})\(X{iNode} * H{iNode} * Z{iNode});
                reg2 = beta * sum(sum(W{iNode} .* W{iNode},2) + eps);
            else
                sumChildW = zeros(d, Cmax);
                for k = 1:nChildInterNode(iNode)
                    sumChildW = sumChildW + W{ChildInterNode{iNode}(k)};
                end
                W{iNode} = (X{iNode} * H{iNode} * X{iNode}' + alpha * D3{iNode} + beta * D2{iNode})\( X{iNode} * H{iNode} * Z{iNode} + beta /nChildInterNode(iNode) * sumChildW);
                temp = W{iNode} - 1/nChildInterNode(iNode) * sumChildW;
                reg2 = beta * sum(sum(temp.*temp,2)+ eps); 
            end
            b{iNode} = 1/sum(d1{iNode}) * ( Z{iNode}' * d1{iNode} - W{iNode}' * (X{iNode} * d1{iNode}));
    
            E1 = X{iNode}' * W{iNode} + ones(nSam(iNode),1) * b{iNode}' - Y{iNode};
            M{iNode} = max(Y{iNode} .* E1,0);                   % Eq. (32)

            E = X{iNode}'*W{iNode} + ones(nSam(iNode),1) * b{iNode}' - Y{iNode} - Y{iNode} .* M{iNode};
            d11 = sqrt(sum(E.*E,2) + eps);
            if iter > floor(0.4*NITER)     % CHANGE FROM 5
                idxe{iNode} = find(d11.^p < epsilon(iNode));
            else
                [temp, ideff] = sort(d11.^p);
                numeff = ceil(0.8 * nSam(iNode));       % ceil()求不小于给定实数的最小整数
                idxe{iNode} = ideff(1:numeff);
                epsilon(iNode) = temp(numeff+1);   % 用前面4份训练数据来确定epsilon的取值
            end;
            d1{iNode} = zeros(nSam(iNode),1) + eps;
            d1{iNode}(idxe{iNode}) = 0.5 * p * d11(idxe{iNode}).^((p-2)/2); D1{iNode} = diag(d1{iNode});   % Eq. (25)
            
            loss(iNode) = sum(d11(idxe{iNode}).^p) + epsilon(iNode) * ( nSam(iNode) - length(idxe{iNode}) );
            reg1 = alpha * sum(sqrt(sum(W{iNode}.*W{iNode},2)));            
            reg(iNode) = reg1 + reg2;                               % 正则项的值
            obj{iNode}(iter,1) = loss(iNode) + reg(iNode);          % 目标函数值
            
            OBJ(iter) = OBJ(iter) + obj{iNode}(iter,1);            
            end
        end
    end
    for j = 1:length(noLeafNode)
        iNode = noLeafNode(j);
        w = sum(W{iNode}.^2, 2);                % sum（X,2）,每一行的和，得到一个列向量
        [~, idx{iNode}] = sort(w, 'descend');   % sort w in a descend order
    end
    if (flag == 1)
        figure;
        set(gcf,'color','w');
        plot(OBJ,'LineWidth',4,'Color',[0 0 1]);
        set(gca,'FontName','Times New Roman','FontSize',11);
        xlabel('Iteration number');
        ylabel('Objective function value');
    end
end

