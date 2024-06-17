function generate_fq_ref(site_cell,half_ext_len,half_fit_range,junc_mode_keyword,output_path)
% site_name, chr, sgRNA_left, arange_left, sgRNA_right, arange_right,
% reverse_seq, reverse_DD
genome_path='~/hg19_with_bowtie2_index/hg19.fa';
[HEADER,SEQ]=fastaread(genome_path);
ref=cell(1,4);
for ii=1:size(site_cell,1)
    nowchr=find(strcmpi(site_cell{ii,2},HEADER));
    sg_left_beg=regexpi(SEQ{nowchr}(site_cell{ii,4}(1):site_cell{ii,4}(2)),site_cell{ii,3})+site_cell{ii,4}(1)-1;
    if strcmpi(SEQ{nowchr}(sg_left_beg-2),'C') && strcmpi(SEQ{nowchr}(sg_left_beg-3),'C')
        left_cut=sg_left_beg+2;
    else
        if strcmpi(SEQ{nowchr}(sg_left_beg+length(site_cell{ii,3})+1),'G') && strcmpi(SEQ{nowchr}(sg_left_beg+length(site_cell{ii,3})+2),'G')
            left_cut=sg_left_beg+length(site_cell{ii,3})-4;
        end
    end
    sg_right_beg=regexpi(SEQ{nowchr}(site_cell{ii,6}(1):site_cell{ii,6}(2)),site_cell{ii,5})+site_cell{ii,6}(1)-1;
    if strcmpi(SEQ{nowchr}(sg_right_beg-2),'C') && strcmpi(SEQ{nowchr}(sg_right_beg-3),'C')
        right_cut=sg_right_beg+2;
    else
        if strcmpi(SEQ{nowchr}(sg_right_beg+length(site_cell{ii,5})+1),'G') && strcmpi(SEQ{nowchr}(sg_right_beg+length(site_cell{ii,5})+2),'G')
            right_cut=sg_right_beg+length(site_cell{ii,5})-4;
        end
    end
    % delete, inverse-forward, inverse-reverse, duplicate
    ref{1}=[upper(SEQ{nowchr}(left_cut-2*half_ext_len+1:left_cut+half_ext_len)),upper(SEQ{nowchr}(right_cut-half_ext_len+1:right_cut+2*half_ext_len))];
    ref{2}=[upper(SEQ{nowchr}(left_cut-2*half_ext_len+1:left_cut+half_ext_len)),upper(seqrcomplement(SEQ{nowchr}(right_cut-2*half_ext_len+1:right_cut+half_ext_len)))];
    ref{3}=[upper(seqrcomplement(SEQ{nowchr}(left_cut-half_ext_len+1:left_cut+2*half_ext_len))),upper(SEQ{nowchr}(right_cut-half_ext_len+1:right_cut+2*half_ext_len))];
    ref{4}=[upper(seqrcomplement(SEQ{nowchr}(left_cut-half_ext_len+1:left_cut+2*half_ext_len))),upper(seqrcomplement(SEQ{nowchr}(right_cut-2*half_ext_len+1:right_cut+half_ext_len)))];
    for jj=1:4
        if site_cell{ii,7}
            ref{jj}=seqrcomplement(ref{jj});
        end
        fid=fopen(fullfile(output_path,[lower(site_cell{ii,1}),'-',lower(junc_mode_keyword{jj}),'.fq.ref']),'w');
        fprintf(fid,"%d\t%d\t%d\n%s\n%d\t%d\t%d\n%d\t%d\t%d\n%s\n%d\t%d\t%d\n",...
            0,0,0,ref{jj}(1:3*half_ext_len),2*half_ext_len,2*half_ext_len-half_fit_range,2*half_ext_len+half_fit_range,...
            4*half_ext_len,4*half_ext_len-half_fit_range,4*half_ext_len+half_fit_range,ref{jj}(3*half_ext_len+1:end),length(ref{jj}),length(ref{jj}),length(ref{jj}));
    end
end