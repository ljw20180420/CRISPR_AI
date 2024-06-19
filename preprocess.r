library(tidyverse)

scoreQuantile <- 0.05
dir("algs") %>%
    map(~ read_tsv(
        file.path("algs", .),
        col_names = c('index', 'count', 'score', 'refId', 'upDangle', 'ref1Start', 'query1Start', 'ref1End', 'query1End', 'randomInsert', 'ref2Start', 'query2Start', 'ref2End', 'query2End', 'downDangle', 'cut1', 'cut2', 'ref', 'query'),
        col_types = "iidiciiiiciiiiciicc",
        col_select = c('count', 'score', 'ref1End', 'ref2Start', 'cut1', 'cut2', 'ref'),
        num_threads = 12,
        lazy = TRUE
    )) |> 
    reduce(bind_rows) |>
    filter(score >= quantile(score, scoreQuantile)) |>
    mutate(
        ref = gsub(pattern = '-', replacement = '', ref),
        ref1Len = str_sub(ref, 2) |> regexpr(pattern = '[acgtn]') + 1
        ref2Start = ref2Start - ref1Len,
        cut2 = cut2 - ref1Len,
        ref1 = toupper(str_sub(ref, 1, ref1Len)),
        ref2 = toupper(str_sub(ref, ref1Len + 1, nchar(ref)))
    ) |>
    summarise(count = sum(count), .by = c(ref1End, ref2Start, cut1, cut2, ref1, ref2)) |>
    write_tsv("data.tsv", col_names = TRUE)
