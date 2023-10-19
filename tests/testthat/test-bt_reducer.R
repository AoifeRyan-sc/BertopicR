#Testing this is actually pretty sophisticated, so first cover the named arguments, then ellipsis separately, then an output.

test_that("bt_make_reducer_umap stops when it's given bad arguments",{

  expect_error(bt_make_reducer_umap(n_neighbours = "15"), "is.numeric.*n_neighb")

  expect_error(bt_make_reducer_umap(n_components = TRUE), "is.numeric.*n_components")
  expect_error(bt_make_reducer_umap(min_dist = "12"))
  expect_error(bt_make_reducer_umap(random_state = "42"), "is.numeric.*random_state")
  expect_error(bt_make_reducer_umap(metric = 2), "is.character.*metric" )
})

test_that("bt_make_reducer_umap's arguments are working as expected", {
  digits <- runif(n = 368*20, min = 0, max = 1)

  #Make an array and a data frame for this test
  my_array <- array(digits, dim = c(20, 368))
  my_df <- as.data.frame(my_array)

  reducer <- bt_make_reducer_umap(n_neighbours = 5L, n_components = 2L, min_dist = 0.001, metric = "cosine", random_state = 32L, verbose = FALSE)

  expect_equal(reducer$min_dist, 0.001)
  expect_equal(reducer$verbose, FALSE)
  expect_equal(reducer$metric, "cosine")
  expect_equal(reducer$n_neighbors, 5L)
  expect_equal(reducer$n_components, 2L)
  expect_equal(reducer$random_state, 32L)
  expect_equal(reducer$low_memory, FALSE)

})

# Similar Tests for bt_make_reducer_pca

test_that("bt_make_reducer_pca stops when given bad arguments", {
  
  expect_error(bt_make_reducer_pca(n_components = "15"), "is.numeric.*n_components")
  expect_error(bt_make_reducer_pca(n_components = 4, svd_solver = "test"), "svd_solver.*is not TRUE")
  expect_error(bt_make_reducer_pca(n_components = 15, test_argument = 5), "Bad argument\\(s\\) attempted to be sent to PCA\\(\\): test_argument")
})

test_that("bt_make_reducer_pca's arguments are working as expected", {
  
  reducer <- bt_make_reducer_pca(n_components = 5, svd_solver = "randomized", whiten = TRUE)
  
  expect_equal(reducer$n_components, 5)
  expect_equal(reducer$svd_solver, "randomized")
  expect_equal(reducer$whiten, TRUE)
  
})

# Similar Tests for by_make_reducer_truncated_svd

test_that("bt_make_reducer_truncated_svd stops when given bad arguments", {
  
  expect_error(bt_make_reducer_truncated_svd(n_components = "15"), "is.numeric.*n_components")
  expect_error(bt_make_reducer_truncated_svd(n_components = 4, svd_solver = "test"),
               "svd_solver.*is not TRUE")
  expect_error(bt_make_reducer_truncated_svd(n_components = 6, n_iter = "5"), "is.numeric.*n_iter")
  expect_error(bt_make_reducer_truncated_svd(n_components = 5, test_argument = 5), 
              regexp = "Bad argument\\(s\\) attempted to be sent to TruncatedSVD\\(\\): test_argument")
})

test_that("bt_make_reducer_truncated_svd's arguments are working as expected", {
  
  reducer <- bt_make_reducer_truncated_svd(n_components = 5, svd_solver = "arpack", 
                                 n_iter = 2, random_state = 42)
  
  expect_equal(reducer$n_components, 5)
  expect_equal(reducer$algorithm, "arpack")
  expect_equal(reducer$random_state, 42)
  expect_equal(reducer$n_iter, 2)
  
})

test_that("bt_empty_reducer's is working as intended - returns a base model", {

  reducer <- bt_empty_reducer()
  #Now check our base model has fit and transform
  expect_true(all(names(reducer) == c("fit", "transform")))

  x <- reducer$fit("xyz")
  expect_true(grepl("^bertopic", class(x)[[1]]))
})

test_that("bt_do_reducing accepts an array or a data frame as embeddings", {
  set.seed(12)
  digits <- runif(n = 368*20, min = 0, max = 1)

  #Make an array and a data frame for this test
  my_array <- array(digits, dim = c(20, 368))
  my_df <- as.data.frame(my_array)

  reducer <- bt_make_reducer_umap(verbose = FALSE)
  reducer_pca <- bt_make_reducer_pca(n_components = 10L)
  reducer_truncated_svd <- bt_make_reducer_pca(n_components = 10L)

  expect_silent(bt_do_reducing(reducer, my_array))
  expect_silent(bt_do_reducing(reducer, my_df))
  expect_silent(bt_do_reducing(reducer_pca, my_df))
  expect_silent(bt_do_reducing(reducer_truncated_svd, my_df))
})

#Integration test with bt_embed
test_that("bt_embed and bt_reducer work together",{
  documents <- bert_example_data %>%
    janitor::clean_names() %>%
    dplyr::filter(!is.na(message)) %>%
    head(10) %>%
    dplyr::pull(message)

  embedder <- bt_make_embedder_st("all-minilm-l6-v2")
  embeddings <- bt_do_embedding(embedder = embedder, documents = documents)
  reducer <- bt_make_reducer_umap( n_neighbours = 2L, verbose = FALSE)
  reducer_pca <- bt_make_reducer_pca(n_components = 10)
  reduced <- bt_do_reducing(reducer = reducer, embeddings = embeddings )
  reduced_pca <- bt_do_reducing(reducer = reducer_pca, embeddings = embeddings)

  expect_true(attributes(reduced)[["reduced"]])
  expect_equal(attributes(reduced)[["original_dim"]], c(10, 384))
  expect_true(attributes(reduced_pca)[["reduced"]])
  expect_equal(attributes(reduced_pca)[["original_dim"]], c(10, 384))
})
