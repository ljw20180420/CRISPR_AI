clc
clear
junc_mode_keyword={'deletion','upstream-inversion','downstream-inversion','duplication'};
jmk={'del','u-i','d-i','dup'};
site_cell={'Beta-globin','chr11','ATTGTTGTTGCCTTGGAGTG',[4093713,4093802],'CTGGTCCCCTGGTAACCTGG',[4094422,4094511],false,false;
    '53.1-KB','chr7','GCTCTTGGTTCACTGATGTG',[27132550,27132609],'AGCGAGATGTTTTCCACCTA',[27185689,27185748],true,false;
    '107-kb','chr7','GCTCTTGGTTCACTGATGTG',[27132550,27132609],'TTGCTGCTTATCGGTTCCTG',[27240327,27240386],true,false;
    '108-kb','chr11','CTGGTCCCCTGGTAACCTGG',[4094437,4094496],'ACCCAATGACCTCAGGCTGT',[4203089,4203148],true,false;
    '115-kb','chr11','ATTGTTGTTGCCTTGGAGTG',[4093728,4093787],'TCACTTGTTAGCGGCATCTG',[4209366,4209425],false,false;
    'CTCF','chr16','GTGTGATTACGCTTGTAGAC',[67660577,67660636],'CAACTTCGTCCCTGCGGCTT',[67662378,67662437],true,true;
    'RAD21','chr8','GCGGCTTGGCTCTTCAATAA',[117864287,117864346],'TAAATTACACTCGAACACAT',[117878838,117878897],false,true;
    'RPA1','chr17','TGGGGGAACACAGTCCAAAG',[1779011,1779070],'GATCTAAAGAGCGGCGGAGT',[1787129,1787188],false,false;
    'ZNF143','chr11','TGTTACGCTGTGCTTGACAG',[9492910,9492969],'GAGTGCATTACAGGCGGTTC',[9499930,9499989],false,false;
    '2.39-Kb-closed-chromatin','chr11','GGAAGATTTGAGGCACTCGG',[50778794,50778853],'TACTTGGAAGCGGGAATATC',[50781183,50781242],false,false;
    '4.15-Kb-closed-chromatin','chr11','GGAAGATTTGAGGCACTCGG',[50778794,50778853],'TGTGTGCATCCGACTCACAT',[50782950,50783009],false,false;
    '54.6-Kb-Open-chromatin-HoxA10','chr7','AGCGAGATGTTTTCCACCTA',[27185689,27185748],'TTGCTGCTTATCGGTTCCTG',[27240327,27240386],true,false;
    '1.16-kb','chr2','CGCTCGGCTCCCTGATTCTA',[234388597,234388656],'AGCAGGAATAAGAGAGAGTC',[234389762,234389821],false,false;
    'PCDH','chr5','GCCACACATCCAAGGCTGAC',[140419754,140419813],'AGATTTGGGGCGTCAGGAAG',[140421026,140421085],false,false;
    'PAMI','chr11','ACCCAATGACCTCAGGCTGT',[4203089,4203148],'TCACTTGTTAGCGGCATCTG',[4209366,4209425],false,false;
    'PAMII','chr2','TCTGTTTTCCTCGCGGTTTC',[176951118,176951177],'GGAGCGCGCTCGCCATCTCC',[176962456,176962515],false,false;
    'PAMIII','chr11','GGAGATGGCAGTGTTGAAGC',[5145456,5145515],'CTAGGGGTCAGAAGTAGTTC',[5226179,5226238],false,false;
    'PAMIV','chr11','TCACTTGTTAGCGGCATCTG',[4209366,4209425],'GGAGATGGCAGTGTTGAAGC',[5145456,5145515],true,false;
    '1.13-MB','chr11','ATTGTTGTTGCCTTGGAGTG',[4093728,4093787],'CTAGGGGTCAGAAGTAGTTC',[5226179,5226238],false,false;
    '1.05-Mb-Open-chromatin-OR52','chr11','CTGGTCCCCTGGTAACCTGG',[4094437,4094496],'GGAGATGGCAGTGTTGAAGC',[5145456,5145515],true,false};

ssn={'Beta','53.1-KB','107-kb','108-kb','115-kb','CTCF','RAD21','RPA1','ZNF143','2.39-Kb','4.15-Kb','54.6-Kb','1.16-kb','PCDH','PAMI','PAMII','PAMIII','PAMIV','1.13-MB','1.05-Mb'};

xlsxs={'/media/ljw/f3b85364-e45d-4166-8db8-1cca425f188e/sx/SJ-LJH-raw-data-for-paper/final-LJH-SJ/LJH/LJH-metadata.xlsx','/media/ljw/f3b85364-e45d-4166-8db8-1cca425f188e/sx/SJ-LJH-raw-data-for-paper/final-LJH-SJ/SJ/SJ_metadata.xlsx'};
raw_4col_all=cell(length(xlsxs),1);
site_cell_on=[site_cell(:,1);{'hoxd';'lcr'}];
for xl=1:length(xlsxs)
    [~,~,raw]=xlsread(xlsxs{xl},2);
    raw=raw(2:end,[2,13]);
    raw_4col_all{xl}=cell(size(raw,1),4);
    nn=0;
    for ii=1:size(raw,1)
        if contains(raw{ii,1},site_cell_on,'IgnoreCase',true) && contains(raw{ii,1},junc_mode_keyword,'IgnoreCase',true)
            nn=nn+1;
            site_name='';
            for jj=1:size(site_cell_on,1)
                if contains(raw{ii,1},site_cell_on{jj,1},'IgnoreCase',true)
                    if ~contains(site_name,site_cell_on{jj,1},'IgnoreCase',true)
                        site_name=site_cell_on{jj,1};
                    end
                end
            end
            raw_4col_all{xl}{nn,1}=site_name;
            for jj=1:length(junc_mode_keyword)
                if contains(raw{ii,1},junc_mode_keyword{jj},'IgnoreCase',true)
                    raw_4col_all{xl}{nn,2}=junc_mode_keyword{jj};
                    break;
                end
            end
            raw_4col_all{xl}{nn,3}=raw{ii,1};
            [startIndex,endIndex]=regexpi(raw_4col_all{xl}{nn,3},raw_4col_all{xl}{nn,1},'once');
            raw_4col_all{xl}{nn,3}(startIndex:endIndex)=[];
            [startIndex,endIndex]=regexpi(raw_4col_all{xl}{nn,3},raw_4col_all{xl}{nn,2},'once');
            raw_4col_all{xl}{nn,3}(startIndex:endIndex)=[];
            raw_4col_all{xl}{nn,3}(isspace(raw_4col_all{xl}{nn,3}))=[];
            raw_4col_all{xl}{nn,4}=raw{ii,2};
            if strcmpi(raw_4col_all{xl}{nn,1},'hoxd')
                raw_4col_all{xl}{nn,1}='PAMII';
            end
            if strcmpi(raw_4col_all{xl}{nn,1},'lcr')
                raw_4col_all{xl}{nn,1}='PAMIII';
            end
            if contains(raw{ii,1},'Beta-globin-930kb','IgnoreCase',true)
                raw_4col_all{xl}{nn,1}='PAMIV';
                [startIndex,endIndex]=regexpi(raw_4col_all{xl}{nn,3},'-930kb','once');
                raw_4col_all{xl}{nn,3}(startIndex:endIndex)=[];
            end
        end
    end
    raw_4col_all{xl}(nn+1:end,:)=[];
    
    already=false(size(raw_4col_all{xl},1),1);
    complete=false(size(raw_4col_all{xl},1),1);
    for ii=1:size(raw_4col_all{xl},1)
        if ~already(ii)
            has_keyword=false(1,4);
            find_rows=[];
            for jj=ii:size(raw_4col_all{xl},1)
                if strcmpi(raw_4col_all{xl}{ii,1},raw_4col_all{xl}{jj,1}) && strcmpi(raw_4col_all{xl}{ii,3},raw_4col_all{xl}{jj,3})
                    find_rows=[find_rows,jj];
                    has_keyword= has_keyword | strcmpi(junc_mode_keyword,raw_4col_all{xl}{jj,2});
                end
            end
            already(find_rows)=true;
            if all(has_keyword)
                complete(find_rows)=true;
            end
        end
    end
    raw_4col_all{xl}(~complete,:)=[];
    raw_4col_all{xl}(:,1:3)=lower(raw_4col_all{xl}(:,1:3));
end
raw_4col_all=vertcat(raw_4col_all{:});
raw_4col_all=sortrows(raw_4col_all,[1,3,2]);

for ra=1:size(raw_4col_all,1)
    tape=site_cell(strcmpi(site_cell(:,1),raw_4col_all{ra,1}),8);
    if tape{1}
        if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{1})
            raw_4col_all{ra,2}=junc_mode_keyword{4};
        else
            if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{4})
                raw_4col_all{ra,2}=junc_mode_keyword{1};
            end
        end
    end
end

fq_paths={'/media/ljw/f3b85364-e45d-4166-8db8-1cca425f188e/sx/Rawdatafq_1434/'};
allPaths={};
for fp=1:length(fq_paths)
    allFiles = dir(fq_paths{fp});
    allPaths = [allPaths;fullfile({allFiles.folder},{allFiles.name})'];
end
mkdir(pwd,'results');
for ra=1:size(raw_4col_all,1)
    [~,basename]=fileparts(raw_4col_all{ra,4});
    flag=false;
    for ap=1:length(allPaths)
        if endsWith(allPaths{ap},['/',raw_4col_all{ra,4}])
            flag=true;
            break;
        end
    end
    if ~flag
        for ap=1:length(allPaths)
            if endsWith(allPaths{ap},['/',raw_4col_all{ra,4},'.gz'])
                flag=true;
                break;
            end
        end
    end
    if ~flag
        error('fq file not found');
    end
    if endsWith(allPaths{ap},'.gz')
        gunzip(allPaths{ap}, fullfile(pwd,'results'));
    else
        while(~copyfile(allPaths{ap}, fullfile(pwd,'results')))
        end
    end
end

cpp_path='~/new_fold/old_desktop/wuqiang/shoujia/projects/Rearrangement/build/rearrangement';
copyfile(cpp_path,fullfile(pwd,'results'));
fq_list=join(raw_4col_all(:,4),',');
fq_list=fq_list{1};
ref_list=cell(size(raw_4col_all,1),1);
for ra=1:size(raw_4col_all,1)
    ref_list{ra}=[raw_4col_all{ra,1},'-',raw_4col_all{ra,2},'.fq.ref'];
end
ref_list=join(ref_list,',');
ref_list=ref_list{1};
fid=fopen('cpp_bash.sh','w');
fprintf(fid,['sudo chmod -R 777 ',fullfile(pwd,'results'),'\ncd ',fullfile(pwd,'results'),'\n./rearrangement -THR_MAX 24 -alg_types local_imbed -files ',fq_list,' -references ',ref_list,'\n']);
fclose(fid);

% run bash cpp_bash.sh

half_fit_range=5;
xx=-half_fit_range:half_fit_range;
for ra=1:size(raw_4col_all,1)
    [~,basename]=fileparts(raw_4col_all{ra,4});
    fileID = fopen(fullfile(pwd,'results',[basename,'.fq.alpha']),'r');
    alpha = fscanf(fileID,'%d %f',[2,inf]);
    fclose(fileID);
    alpha(1,:)=[];
    fileID = fopen(fullfile(pwd,'results',[basename,'.fq.beta']),'r');
    beta = fscanf(fileID,'%d %f',[2,inf]);
    fclose(fileID);
    beta(1,:)=[];
    if ra==1 || ~strcmpi(raw_4col_all{ra,1},raw_4col_all{ra-1,1}) || ~strcmpi(raw_4col_all{ra,2},raw_4col_all{ra-1,2}) || ~strcmpi(raw_4col_all{ra,3},raw_4col_all{ra-1,3})
        alpha_all=alpha;
        beta_all=beta;
        count=1;
    else
        alpha_all=alpha_all+alpha;
        beta_all=beta_all+beta;
        count=count+1;
    end
    if ra==size(raw_4col_all,1) || ~strcmpi(raw_4col_all{ra,1},raw_4col_all{ra+1,1}) || ~strcmpi(raw_4col_all{ra,2},raw_4col_all{ra+1,2}) || ~strcmpi(raw_4col_all{ra,3},raw_4col_all{ra+1,3})
        alpha_all=alpha_all/count;
        beta_all=beta_all/count;
        tape=jmk(strcmpi(junc_mode_keyword,raw_4col_all{ra,2}));
        fileID = fopen(fullfile(pwd,'results',[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',tape{1},'.alpha']),'w');
        for xa=1:length(xx)
            fprintf(fileID,'%d\t%f\n',xx(xa),alpha_all(xa));
        end
        fclose(fileID);
        fileID = fopen(fullfile(pwd,'results',[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',tape{1},'.beta']),'w');
        for xa=1:length(xx)
            fprintf(fileID,'%d\t%f\n',xx(xa),beta_all(xa));
        end
        fclose(fileID);
    end
end

% fid=fopen('cpp_bash_FL.sh','w');
% fprintf(fid,['sudo chmod -R 777 ',fullfile(pwd,'results'),'\ncd ',fullfile(pwd,'results'),'\n./crispr_map -THR_MAX 24 -fq_list ']);
% alpha_list=cell(size(raw_4col_all,1),1);
% beta_list=cell(size(raw_4col_all,1),1);
% for ra=1:size(raw_4col_all,1)
%     [~,basename]=fileparts(raw_4col_all{ra,4});
%     tape=site_cell(strcmpi(site_cell(:,1),raw_4col_all{ra,1}),7);
%     if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{2})
%         if ~tape{1}
%             alpha_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{1},'.alpha'];
%             beta_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{4},'.beta'];
%         else
%             alpha_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{4},'.alpha'];
%             beta_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{1},'.beta'];
%         end
%     else
%         if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{3})
%             if ~tape{1}
%                 alpha_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{4},'.alpha'];
%                 beta_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{1},'.beta'];
%             else
%                 alpha_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{1},'.alpha'];
%                 beta_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{4},'.beta'];
%             end
%         else
%             if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{1})
%                 if ~tape{1}
%                     alpha_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{2},'.alpha'];
%                     beta_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{3},'.beta'];
%                 else
%                     alpha_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{3},'.alpha'];
%                     beta_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{2},'.beta'];
%                 end
%             else
%                 if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{4})
%                     if ~tape{1}
%                         alpha_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{3},'.alpha'];
%                         beta_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{2},'.beta'];
%                     else
%                         alpha_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{2},'.alpha'];
%                         beta_list{ra}=[raw_4col_all{ra,1},raw_4col_all{ra,3},'_',jmk{3},'.beta'];
%                     end
%                 end
%             end
%         end
%     end
% end
% for ra=1:size(raw_4col_all,1)
%     fprintf(fid,raw_4col_all{ra,4});
%     if ra~=size(raw_4col_all,1)
%         fprintf(fid,',');
%     end
% end
% fprintf(fid,' -ini_alpha ');
% for ra=1:length(alpha_list)
%     fprintf(fid,alpha_list{ra});
%     if ra~=length(alpha_list)
%         fprintf(fid,',');
%     end
% end
% fprintf(fid,' -ini_beta ');
% for ra=1:length(beta_list)
%     fprintf(fid,beta_list{ra});
%     if ra~=length(beta_list)
%         fprintf(fid,',');
%     end
% end
% fclose(fid);

% run bash cpp_bash_FL.sh

mode={'del-dup','u-i-d-i'};
FZ=10;
half_fit_range=5;
xx=-half_fit_range:half_fit_range;
thx=0.02;
thy=0.02;
mks=0.005;
hei=(1-2*thy)/7/2;
wid=(1-thx-3*(2*thx+mks))/12;
h_fig=cell(2,6);
h_axes=cell(size(h_fig));
for rf=1:size(h_fig,1)
    for cf=1:size(h_fig,2)
        h_fig{rf,cf}=figure('Position',[0 0 1000 1000]);
        h_axes{rf,cf}=axes(h_fig{rf,cf},'Position',[0,0,1,1],'Visible','off');
        if rf==1
            x_text='real distribution';
            y_text='restored distribution';
            for co=1:3
                for coco=1:4
                    text(h_axes{rf,cf},(2*thx+mks+4*wid)*(co-1)+3*thx+mks+wid*(coco-0.5),1-thy/2,jmk{coco},'HorizontalAlignment','center','Fontname', 'Times New Roman','Fontsize',FZ);
                end
            end
        else
            x_text='relative cleavage';
            y_text='probability';
            for co=1:3
                for coco=1:4
                    text(h_axes{rf,cf},(2*thx+mks+4*wid)*(co-1)+3*thx+mks+wid*(coco-0.5),1-thy/2,['end ',num2str(coco)],'HorizontalAlignment','center','Fontname', 'Times New Roman','Fontsize',FZ);
                end
            end
        end
        text(h_axes{rf,cf},0.5,thy/2,x_text,'HorizontalAlignment','center','Fontname', 'Times New Roman','Fontsize',FZ);
        text(h_axes{rf,cf},thx/2,0.5,y_text,'HorizontalAlignment','center','Rotation',90,'Fontname', 'Times New Roman','Fontsize',FZ);
    end
end

total_num=0;
allFiles = dir(fullfile(pwd,'results'));
allPaths = fullfile({allFiles.folder},{allFiles.name});
for ra=1:size(raw_4col_all,1)
    [~,basename]=fileparts(raw_4col_all{ra,4});
    if ra==1 || ~strcmpi(raw_4col_all{ra,1},raw_4col_all{ra-1,1}) || ~strcmpi(raw_4col_all{ra,3},raw_4col_all{ra-1,3})
        dis_index=cell(4,2);
        dis_real=cell(4,2);
        dis_model=cell(4,2);
        dis_count=zeros(4,2);
        dis_multi_index=cell(4,2);
        dis_multi_real=cell(4,2);
        dis_multi_model=cell(4,2);
        dis_multi_count=zeros(4,2);
        distri=zeros(4,2*half_fit_range+1,4);
        count=zeros(4,1,4);
    end
    for md=1:length(mode)
        if (md==1 && (strcmpi(raw_4col_all{ra,2},junc_mode_keyword{1}) || strcmpi(raw_4col_all{ra,2},junc_mode_keyword{4})))...
                || (md==2 && (strcmpi(raw_4col_all{ra,2},junc_mode_keyword{2}) || strcmpi(raw_4col_all{ra,2},junc_mode_keyword{3})))
            fileID = fopen(fullfile(pwd,'results',[basename,'.fq.box']),'r');
            dis = fscanf(fileID,'%d %f %f',[3,inf]);
            fclose(fileID);
            fileID = fopen(fullfile(pwd,'results',[basename,'.fq.box.multi']),'r');
            dis_multi = fscanf(fileID,'%d %f %f %d %d',[5,inf]);
            fclose(fileID);
            fileID = fopen(fullfile(pwd,'results',[basename,'.fq.alpha']),'r');
            alpha = fscanf(fileID,'%d %f',[2,inf]);
            fclose(fileID);
            alpha(1,:)=[];
            fileID = fopen(fullfile(pwd,'results',[basename,'.fq.beta']),'r');
            beta = fscanf(fileID,'%d %f',[2,inf]);
            fclose(fileID);
            beta(1,:)=[];
        else
            targetPaths=allPaths(endsWith(allPaths,[basename,'.fq.box']));
            if length(targetPaths)>2
                targetPaths=targetPaths(endsWith(targetPaths,strcat({'/','_'},[basename,'.fq.box'])));
                if length(targetPaths)>2
                    error('find too much');
                end
            end
            [~,maxI]=max(strlength(targetPaths));
            fileID = fopen(targetPaths{maxI},'r');
            dis = fscanf(fileID,'%d %f %f',[3,inf]);
            fclose(fileID);
            targetPaths=allPaths(endsWith(allPaths,[basename,'.fq.box.multi']));
            if length(targetPaths)>2
                targetPaths=targetPaths(endsWith(targetPaths,strcat({'/','_'},[basename,'.fq.box.multi'])));
                if length(targetPaths)>2
                    error('find too much');
                end
            end
            [~,maxI]=max(strlength(targetPaths));
            fileID = fopen(targetPaths{maxI},'r');
            dis_multi = fscanf(fileID,'%d %f %f %d %d',[5,inf]);
            fclose(fileID);
            targetPaths=allPaths(endsWith(allPaths,[basename,'.fq.alpha']));
            if length(targetPaths)>2
                targetPaths=targetPaths(endsWith(targetPaths,strcat({'/','_'},[basename,'.fq.alpha'])));
                if length(targetPaths)>2
                    error('find too much');
                end
            end
            [~,maxI]=max(strlength(targetPaths));
            fileID = fopen(targetPaths{maxI},'r');
            alpha = fscanf(fileID,'%d %f',[2,inf]);
            fclose(fileID);
            alpha(1,:)=[];
            targetPaths=allPaths(endsWith(allPaths,[basename,'.fq.beta']));
            if length(targetPaths)>2
                targetPaths=targetPaths(endsWith(targetPaths,strcat({'/','_'},[basename,'.fq.beta'])));
                if length(targetPaths)>2
                    error('find too much');
                end
            end
            [~,maxI]=max(strlength(targetPaths));
            fileID = fopen(targetPaths{maxI},'r');
            beta = fscanf(fileID,'%d %f',[2,inf]);
            fclose(fileID);
            beta(1,:)=[];
        end

        if sum(alpha)>0 && sum(beta)>0
            for jm=1:length(junc_mode_keyword)
                if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{jm})
                    if isempty(dis_model{jm,md})
                        dis_index{jm,md}=dis(1,:);
                        dis_real{jm,md}=dis(2,:);
                        dis_model{jm,md}=dis(3,:);
                        dis_multi_index{jm,md}=dis_multi(5,:);
                        dis_multi_real{jm,md}=dis_multi(2,:);
                        dis_multi_model{jm,md}=dis_multi(3,:);
                    else
                        dis_real{jm,md}=dis_real{jm,md}+dis(2,:);
                        dis_multi_real{jm,md}=dis_multi_real{jm,md}+dis_multi(2,:);
                        dis_model{jm,md}=dis_model{jm,md}+dis(3,:);
                        dis_multi_model{jm,md}=dis_multi_model{jm,md}+dis_multi(3,:);
                    end
                    dis_count(jm,md)=dis_count(jm,md)+1;
                    dis_multi_count(jm,md)=dis_multi_count(jm,md)+1;
                    break;
                end
            end
            
            mdd=2*md-2;
            if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{1})
                distri(1,:,1+mdd)=distri(1,:,1+mdd)+alpha;
                count(1,1,1+mdd)=count(1,1,1+mdd)+1;
                distri(4,:,1+mdd)=distri(4,:,1+mdd)+beta;
                count(4,1,1+mdd)=count(4,1,1+mdd)+1;
            else
                if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{2})
                    distri(1,:,2+mdd)=distri(1,:,2+mdd)+alpha;
                    count(1,1,2+mdd)=count(1,1,2+mdd)+1;
                    distri(3,:,2+mdd)=distri(3,:,2+mdd)+flip(beta);
                    count(3,1,2+mdd)=count(3,1,2+mdd)+1;
                else
                    if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{3})
                        distri(2,:,2+mdd)=distri(2,:,2+mdd)+flip(alpha);
                        count(2,1,2+mdd)=count(2,1,2+mdd)+1;
                        distri(4,:,2+mdd)=distri(4,:,2+mdd)+beta;
                        count(4,1,2+mdd)=count(4,1,2+mdd)+1;
                    else
                        if strcmpi(raw_4col_all{ra,2},junc_mode_keyword{4})
                            distri(2,:,1+mdd)=distri(2,:,1+mdd)+flip(alpha);
                            count(2,1,1+mdd)=count(2,1,1+mdd)+1;
                            distri(3,:,1+mdd)=distri(3,:,1+mdd)+flip(beta);
                            count(3,1,1+mdd)=count(3,1,1+mdd)+1;
                        end
                    end
                end
            end
        end
    end
    
    if (ra==size(raw_4col_all,1) || ~strcmpi(raw_4col_all{ra,1},raw_4col_all{ra+1,1}) || ~strcmpi(raw_4col_all{ra,3},raw_4col_all{ra+1,3})) && all(count(:)>0)
        total_num=total_num+1;
        fig_num=ceil(total_num/21);
        pos_num=mod(total_num-1,21)+1;
        for da=1:numel(dis_real)
            dis_real{da}=dis_real{da}/dis_count(da);
            dis_multi_real{da}=dis_multi_real{da}/dis_multi_count(da);
        end
        for da=1:numel(dis_model)
            dis_model{da}=dis_model{da}/dis_count(da);
            dis_multi_model{da}=dis_multi_model{da}/dis_multi_count(da);
        end
        
        distri=distri./count;
        tape=site_cell(strcmpi(site_cell(:,1),raw_4col_all{ra,1}),7);
        if tape{1}
            for md=1:2
                mdd=2*md-2;
                distri(:,:,1+mdd)=rot90(distri(:,:,1+mdd),2);
                tape=distri(1,:,2+mdd);
                distri(1,:,2+mdd)=distri(3,:,2+mdd);
                distri(3,:,2+mdd)=tape;
                tape=distri(2,:,2+mdd);
                distri(2,:,2+mdd)=distri(4,:,2+mdd);
                distri(4,:,2+mdd)=tape;
            end
        end
        
        junc_pair=[1,2;4,3;4,2;1,3];
        [tpos,ppos]=plot_pos(pos_num,thx,thy,hei,wid,mks);
        tape=ssn(strcmpi(site_cell(:,1),raw_4col_all{ra,1}));
        tape2=split(raw_4col_all{ra,3},'-');
        tape2(strcmpi(tape2,''))=[];
        tape2=join(tape2,'-');
        text(h_axes{1,fig_num},tpos(1,1),tpos(1,2),tape{1},'HorizontalAlignment','center','Rotation',90,'Fontname', 'Times New Roman','Fontsize',FZ);
        text(h_axes{1,fig_num},tpos(2,1),tpos(2,2),tape2{1},'HorizontalAlignment','center','Rotation',90,'Fontname', 'Times New Roman','Fontsize',FZ);
        text(h_axes{2,fig_num},tpos(1,1),tpos(1,2),tape{1},'HorizontalAlignment','center','Rotation',90,'Fontname', 'Times New Roman','Fontsize',FZ);
        text(h_axes{2,fig_num},tpos(2,1),tpos(2,2),tape2{1},'HorizontalAlignment','center','Rotation',90,'Fontname', 'Times New Roman','Fontsize',FZ);
        for fc=1:4
            for md=1:2
                kk=4*(md-1)+fc;
                mdd=2*md-2;
                axdis=axes(h_fig{1,fig_num},'Position',ppos(kk,:));
                tape=~ismember(dis_index{fc,md},dis_multi_index{fc,md});
                plot(axdis,[0,1],[0,1],'--b',dis_real{fc,md}(tape),dis_model{fc,md}(tape),'ok',dis_multi_real{fc,md},dis_multi_model{fc,md},'xr');
                set(axdis,'xtick',[],'ytick',[],'XLim',[0,1],'YLim',[0,1]);
                
                axab=axes(h_fig{2,fig_num},'Position',ppos(kk,:));
                plot(axab,xx,distri(fc,:,1+mdd),'-k',xx,distri(fc,:,2+mdd),'--k');
                set(axab,'xtick',[],'ytick',[],'XLim',[-5,5],'YLim',[0,1]);
            end
        end
    end
end
for cf=1:6
    print('-painters',h_fig{1,cf},fullfile(pwd,'figures',['restore_distribution_',num2str(cf),'.eps']),'-depsc');
    print('-painters',h_fig{1,cf},fullfile(pwd,'figures',['restore_distribution_',num2str(cf),'.jpg']),'-djpeg');
    print('-painters',h_fig{2,cf},fullfile(pwd,'figures',['junction_consist_',num2str(cf),'.eps']),'-depsc');
    print('-painters',h_fig{2,cf},fullfile(pwd,'figures',['junction_consist_',num2str(cf),'.jpg']),'-djpeg');
end