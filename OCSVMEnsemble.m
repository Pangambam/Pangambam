clear all
close all
cd 'D:\Programs\Matlab R2019a\Workspace\SemiSupervised\OCE';
load('BOTSWANA.mat');
LabData = normalize(reshape(DATA,size(DATA,1)*size(DATA,2),size(DATA,3)),'range');
LabGT = reshape(GT,size(DATA,1)*size(DATA,2),1);
count = 1;
Spatial = zeros(size(DATA,1)*size(DATA,2),2);
for c = 1:size(DATA,2)
    for r = 1:size(DATA,1)
        Spatial(count,:) = [r,c];
        count = count+1;
    end
end
nonZeroIDX = LabGT~=0;
LabData = LabData(nonZeroIDX,:);
LabGT = LabGT(nonZeroIDX,:);
Spatial = Spatial(nonZeroIDX,:);
rng(size(LabGT,1));
IDX = randperm(size(LabGT,1))';
LabData = LabData(IDX,:);
LabGT = LabGT(IDX,:);
Spatial = Spatial(IDX,:);
clear c; clear count, clear nonZeroIDX; clear r; clear IDX;

%DIMENSION REDUCTION ALONG SPECTRAL DOMAIN BY DATA FUSION
numPart = 10; %NUMBER OF SEGMENTS
numBand = floor(size(LabData,2)/numPart); %NUMBER OF BANDS PER PARTITION
Interval =(1:numBand:size(LabData,2));
for i =2:size(Interval,2)
    IDX(i-1,:) = [Interval(i-1),Interval(i)-1];
end
IDX(i,:) = [1+ IDX(i-1,2), size(LabData,2)];
RedData = zeros(size(LabData,1),size(IDX,1));
for i = 1:size(IDX,1)
    PartData = (LabData(:,IDX(i,1):IDX(i,2)));
    PartData = transpose(PartData);
    FusedData = mean(PartData);
    RedData(:,i) = FusedData';
    clear PartData; clear FusedData;
end
clear i;
%%
%CROSS VALIDATION
cumTarget = cell(10,1);
cumOutput = cell(10,1);
CVIDX = crossvalind('Kfold',LabGT,10);
for cvFold = 1:10
    
    %cvFold =1;
    fprintf('CrossVal. Fold %d started. \n',cvFold);
    pause(1);
    TestIDX = (CVIDX == cvFold);
    TrainIDX = ~TestIDX;
    TestDATA = RedData(TestIDX,:);
    TestGT = LabGT(TestIDX,:);
    cumTarget(cvFold,1) = {TestGT};
    TrainData = RedData(TrainIDX,:);
    TrainGT = LabGT(TrainIDX,:);
    rng(cvFold);
    IDX = randperm(size(TrainData,1));
    SplitIDX = round(0.4*size(TrainData,1));
    LTrainData = TrainData(1:SplitIDX,:);
    LTrainGT = TrainGT(1:SplitIDX,:);
    SpaTrain = Spatial(1:SplitIDX,:);
    UTrainData = TrainData(1+SplitIDX:size(TrainData,1),:);
    fprintf('ONE CLASS SVM LEARNING!!! \n');
    %%
    [Modl, numClass] = TrainModel(LTrainData, LTrainGT);
    
    %%
    %SEMISUPERVISED (LABELING THE UNLABELED SAMPLES)
    %FIND INITIAL CENTRES OF EACH CLASS FROM THE LABELED DATA
    fprintf('FINDING CLASSWISE SPECTRAL CENTRES. \n');
    Centers = cell(numClass,1);
    ClassWiseData = groupByClass(LTrainData,LTrainGT);
    %%
    for i = 1:numClass
        Centers(i) = {subclust(ClassWiseData{i,1},0.4)};
        [~, Center] = kmeans(ClassWiseData{i,1},size(Centers{i,1},1),'Start',Centers{i,1});
        %[~, Center] = kmeans(ClassWiseData{i,1},5);
        Centers(i) = {Center};
        clear Center;
    end
    %%
    LabCount = 0;
    fprintf('SEMISUPERVISED LABELING STARTED!!! \n');
    fprintf('CALCULATE SPECTRAL DISTANCE FROM CLASS CENTRES AND TAKE THE MIN DIST FOR UNLABELLED SAMPLES!!!\n');
    for unCount = 1:size(UTrainData,1)
        currUData = UTrainData(unCount,:);
        %CALCULATE SPECTRAL DISTANCE FROM CLASS CENTRES AND TAKE THE MIN DIST
        DIST = cell(size(Modl,1),1);
        MINDIST = zeros(size(Modl,1),1);
        for mCount = 1:size(Modl,1)
            ModCenter = Centers{mCount,1};
            ModCenDIST = zeros(size(ModCenter,1),1);
            for cenCount = 1:size(ModCenter,1)
                ModCenDIST(cenCount) = pdist([currUData;ModCenter(cenCount,:)]);
            end
            DIST(mCount) = {ModCenDIST};
            MINDIST(mCount,1) = min(ModCenDIST);
            clear ModCenDIST;
        end
        [~,minIDX] = sort(MINDIST);
        minIDX = minIDX(1);
        %PREDICT THE UNLABELED SAMPLE WITH NEAREST OC-SVM
        %IF IT IS 1 THEN INCLUDE
        for i=1:numClass
            [~,scorePred(i)] = predict(Modl{i},currUData);
        end
        [~,maxScoreIDX] = sort(scorePred); maxScoreIDX=maxScoreIDX(i);
        
        %Global Decision if maxScoreIDX=minIDX;
        if (maxScoreIDX==minIDX)
            SpecClass =maxScoreIDX;
        else
            SpecClass = 1000;
        end
        
        %predLabelByMinModl = predict(Modl{minIDX,1},currUData);
        
        
        %SPATIAL
        [IDX,~] = find(currUData == RedData); IDX = IDX(1);
        UPosition = Spatial(IDX,:);
        [nearIDX,~] = knnsearch(SpaTrain, UPosition, 'k', 25);
        nearSamples = SpaTrain(nearIDX,:);
        
        LabNearSamples = zeros(size(nearSamples,1),1);
        for nearSCount = 1:size(nearSamples,1)
            LabNearSamples(nearSCount,1) = GT(nearSamples(nearSCount,1),nearSamples(nearSCount,2));
        end
        nearClasses = unique(LabNearSamples);
        for i=1:size(nearClasses,1)
            IDX = LabNearSamples==nearClasses(i);
            nearSpaCoOr = nearSamples(IDX,:);
            SpaScore(i) = 0;
            for j=1:size(nearSpaCoOr,1)
                X = [UPosition;nearSpaCoOr(j,:)];
                SpaScore(i) = SpaScore(i)+(1/pdist(X));
            end
        end
        [~,IDX]=sort(SpaScore,'DESC'); IDX=IDX(1); clear SpaScore;
        SpaClass = nearClasses(IDX);
        
        if (SpecClass==SpaClass)
            ClassWiseData{SpaClass,1}=[ClassWiseData{SpaClass,1};currUData];
            LabCount = LabCount + 1;
        end
        %%
        
        %UPDATE THE CLUSTER CENTRES and BSVMs BATCHWISE
        if mod(LabCount,round(size(UTrainData,1)/10)) == 0
			fprintf('UPDATING CLUSTERS AND SVMS BATCHWISE. \n');
            newGT = giveLabel(ClassWiseData);
            newTData = cell2mat(ClassWiseData);
            [Modl, numClass] = TrainModel(newTData, newGT);
            %Centers = cell(numClass,1);
            for i = 1:numClass
                rng(i);
                [~, Center] = kmeans(ClassWiseData{i,1},size(Centers{i,1},1),'Start',Centers{i,1});
                Centers(i) = {Center};
            end
            fprintf('UPDATED SUCCESSFULLY. \n');
        end
    end
    
    
    %%
    %ORIGINAL + NEW LABEL
    ClassWiseLabel = cell(numClass,1);
    for numClass=1:size(ClassWiseData,1)
        ClassWiseLabel{numClass,1} = numClass*ones (size(ClassWiseData{numClass,1},1),1);
    end
    
    %%
    %%KNN
    fprintf('SUPERVISED CLASSIFICATION AND TESTING!!!. \n');
    NewTrainData = cell2mat(ClassWiseData);
    NewTrainLabel = categorical(cell2mat(ClassWiseLabel));
    Mdl = fitcknn(NewTrainData,NewTrainLabel,'NumNeighbors',1);
    PREDICTED = predict(Mdl,TestDATA);
    cumOutput(cvFold,1) = {PREDICTED};
end
%%
OUTPUT = cell(cvFold,1);
for i =1:cvFold
    OUTPUT{i,1} = double(cumOutput{i,1});
end
TARGET = categorical(cell2mat(cumTarget));
OUTPUT = categorical(cell2mat(OUTPUT));
CMAT = confusionmat(TARGET,OUTPUT);
save('BOTSWANA40.mat','CMAT');
plotconfusion(TARGET,OUTPUT);



%%
%WRITE THE RESULT
Acc = cell(10,1);
overAcc = cell(10,1);
for i=1:10
    currOut = cumOutput{i,1};
    currTar = categorical(cumTarget{i,1});
    currCM = confusionmat(currTar,categorical(currOut));
    for numClass=1:size(currCM,1)
        currTP = currCM(numClass,numClass); currTP(currTP==0)=1;
        tempMat = currCM;
        tempMat(numClass,:) = []; tempMat(:,numClass) = [];
        currTN = sum(sum(tempMat));
        currTN(currTN==0)=1;
        currFP = sum(currCM(numClass,:))-currCM(numClass,numClass);
        currFP(currFP==0)=1;
        currFN = sum(currCM(:,numClass))-currCM(numClass,numClass);
        currFN(currFN==0)=1;
        CM(i,numClass) = {[currTP,currFP;currFN,currTN]};
        PRECISION(i,numClass) = currTP/(currTP+currFP);
        RECALL(i,numClass) = currTP/(currTP+currFN);
        F(i,numClass) = 2*(PRECISION(i,numClass)*RECALL(i,numClass))/(PRECISION(i,numClass)+RECALL(i,numClass));
    end
    SUM = sum(currCM); SUM(SUM==0)=1;
    overAcc(i,1) = {100*(sum(diag(currCM))/sum(sum(currCM)))};
    currAcc = 100*(diag(currCM)'./SUM);
    Acc(i,1) = {currAcc};
end
Acc=cell2mat(Acc);
accCLASSWISE=mean(Acc);
overAcc=cell2mat(overAcc);
accCLASSWISE = accCLASSWISE';
OverACC = mean(overAcc)*ones(size(accCLASSWISE));
AvClasswiseAcc = mean(accCLASSWISE)*ones(size(accCLASSWISE));
AvePRECISION = mean(mean(PRECISION))*ones(size(accCLASSWISE));
AveRECALL = mean(mean(RECALL))*ones(size(accCLASSWISE));
AveF = mean(mean(F))*ones(size(accCLASSWISE));
RESULT = [array2table(accCLASSWISE),array2table(AvClasswiseAcc),...
    array2table(OverACC),array2table(AvePRECISION),...
    array2table(AveRECALL),array2table(AveF)];
writetable(RESULT,'BOTSWANA40.csv');
%%






function [Modl, numClass] = TrainModel(LTrainData, LTrainGT)
numClass = size(unique(LTrainGT),1);
Modl =  cell(numClass,1);
ClassWiseTrData = cell(numClass,1);
for i=1:numClass
    inIDX = LTrainGT==i;
    outIDX = ~inIDX;
    inData = LTrainData(inIDX,:);
    ClassWiseTrData(i) = {inData};
    outData = LTrainData(outIDX,:);
    rng(size(inData,1));
    rIDX = randperm(size(outData,1),round(size(inData,1)*0.1));
    outlier = outData(rIDX,:);
    currOCTrData = [inData;outlier];
    currOCGT = ones(size(currOCTrData,1),1);
    rng(1);
    currOCSVMModl = fitcsvm(currOCTrData,currOCGT,'KernelScale','auto','Standardize',true,...
        'OutlierFraction',0.1);
    Modl(i)= {currOCSVMModl};
    clear currOCSVMModl; clear currOCTrData; clear currOCGT;
end
end

function ClassWiseData = groupByClass(LTrainData,LTrainGT)
numClass = size(unique(LTrainGT),1);
ClassWiseData = cell(numClass,1);
for i = 1:numClass
    inIDX = LTrainGT==i;
    inData = LTrainData(inIDX,:);
    ClassWiseData(i) = {inData};
end
end

function ClassWiseLabel = giveLabel(ClassWiseDataCell)
ClassWiseLabel = cell(size(ClassWiseDataCell,1),1);
for numClass=1:size(ClassWiseDataCell,1)
    ClassWiseLabel{numClass,1} = numClass*ones (size(ClassWiseDataCell{numClass,1},1),1);
end
ClassWiseLabel = cell2mat(ClassWiseLabel);
end