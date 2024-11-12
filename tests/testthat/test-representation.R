test_that("bt_representation_keybert errors on bad input", {

  bt <- reticulate::import("bertopic")
  docs <- stringr::sentences[1:10]
  model <- bt$BERTopic(embedding_model = bt_empty_embedder(),
                       umap_model = bt_empty_reducer())$fit(docs, array(runif(10), dim = c(10, 1)))
  
  expect_error(bt_representation_keybert(top_n_words = "test"), "is.numeric.*top_n_words")
  expect_error(bt_representation_keybert(nr_repr_docs = "test"), "is.numeric.*nr_repr_docs")
  expect_error(bt_representation_keybert(nr_samples = "test"), "is.numeric.*nr_samples")
  expect_error(bt_representation_keybert(nr_candidate_words = "test"), "is.numeric.*nr_candidate_words")
  expect_error(bt_representation_keybert(documents = 6), "is.character.*documents")
  expect_error(bt_representation_keybert(documents = "test",
                                         document_embeddings = "test"), "is.numeric.*document_embeddings")
  expect_error(bt_representation_keybert(documents = "test",
                                         document_embeddings = 5), "is.array.*is.data.frame")
  expect_error(bt_representation_keybert(documents = "test",
                                         document_embeddings = array(runif(10), dim = c(5, 2)),
                                         fitted_model =" test"), "Model should be a BERTopic model")
  expect_error(bt_representation_keybert(documents = "test",
                                         document_embeddings = array(runif(10), dim = c(5, 2)),
                                         fitted_model = bt$BERTopic()), 
               regexp = "BERTopic model is not fitted, use bt_fit_model to fit.")
  expect_error(bt_representation_keybert(documents = "test",
                                         document_embeddings = array(runif(10), dim = c(5, 2)),
                                         fitted_model = model,
                                         embedding_model = "embedding model"), 
               regexp = "This package currently only supports embedding models from the sentence transformer library, embedder should be a sentence transformers model")
  expect_error(bt_representation_keybert(documents = "test",
                                         document_embeddings = array(runif(10), dim = c(5, 2)),
                                         fitted_model = model,
                                         embedding_model = bt_make_embedder_st("all-miniLM-L6-v2")),
               ".*Number of documents:.*Number of embeddings:.*")
               
})

test_that("bt_representation_keybert returns correct output on correct input", {
  
  bt <- reticulate::import("bertopic")
  docs <- stringr::sentences[1:10]
  embeddings <- array(runif(3840), dim = c(10, 384))
  model <- bt$BERTopic(embedding_model = bt_empty_embedder(),
                       umap_model = bt_empty_reducer())$fit(docs, embeddings)
  embedder <- bt_make_embedder_st("all-MiniLM-L6-v2")
  
  representation_keybert <- bt_representation_keybert(fitted_model = model,
                                                      documents = docs,
                                                      document_embeddings = embeddings,
                                                      embedding_model = embedder)
  nr_topics <- model$get_topic_info() %>% nrow()
  expect_true("-1" %in% names(representation_keybert))
  expect_equal(nr_topics, length(representation_keybert))
  expect_true(is.character(unlist(representation_keybert)))
})

test_that("bt_representation_mmr errors on bad input", {
  
  # fitted_model,
  # embedding_model,
  # diversity = 0.1,
  # top_n_words = 10
  bt <- reticulate::import("bertopic")
  docs <- stringr::sentences[1:10]
  embeddings <- array(runif(10), dim = c(10,1))
  model <- bt$BERTopic(embedding_model = bt_empty_embedder(),
                       umap_model = bt_empty_reducer())$fit(docs, embeddings)
  
  expect_error(bt_representation_mmr(diversity = "test"), "is.numeric.*diversity")
  expect_error(bt_representation_mmr(diversity = 1.2), "diversity <=")
  expect_error(bt_representation_mmr(top_n_words = "test"), "is.numeric.*top_n_words")
  expect_error(bt_representation_mmr(fitted_model = "test"), "Model should be a BERTopic model")
  expect_error(bt_representation_mmr(fitted_model = bt$BERTopic()), "BERTopic model is not fitted, use bt_fit_model to fit.")
  expect_error(bt_representation_mmr(fitted_model = model,
                                     embedding_model = "test"),
               "This package currently only supports embedding models from the sentence transformer library, embedder should be a sentence transformers model")
  })

test_that("bt_representation_mmr returns correct output on correct input", {
  
  bt <- reticulate::import("bertopic")
  docs <- stringr::sentences[1:10]
  embeddings <- array(runif(3840), dim = c(10,384))
  model <- bt$BERTopic(embedding_model = bt_empty_embedder(),
                       umap_model = bt_empty_reducer())$fit(docs, embeddings)
  
  representation_mmr <- bt_representation_mmr(fitted_model = model,
                                              embedding_model = bt_make_embedder_st("all-MiniLM-L6-v2"))
  nr_topics <- model$get_topic_info() %>% nrow()
  expect_true("-1" %in% names(representation_mmr))
  expect_equal(nr_topics, length(representation_mmr))
  expect_true(is.character(unlist(representation_mmr)))
})

test_that("bt_representation_openai errors on bad input", {
  bt <- reticulate::import("bertopic")
  unfitted_model <- bt$BERTopic()
  docs <- stringr::sentences[1:10]
  embeddings <- array(runif(10), dim = c(10,1))
  model <- bt$BERTopic(embedding_model = bt_empty_embedder(),
                       umap_model = bt_empty_reducer())$fit(docs, embeddings)
  
  expect_error(bt_representation_openai(openai_model = "gpt",
                                        documents = 5), "is.character.*documents.")
  expect_error(bt_representation_openai(documents = "test",
                                        openai_model = 6), "is.character.*openai_model.")
  expect_error(bt_representation_openai(openai_model = "gpt", 
                                        documents = "test",
                                        chat = "test"), "is.logical.*chat")
  expect_error(bt_representation_openai(openai_model = "gpt", 
                                        documents = "test",
                                        nr_repr_docs = "test"), "is.numeric.*nr_repr_docs")
  expect_error(bt_representation_openai(openai_model = "gpt", 
                                        documents = "test",
                                        api_key = "test"), "str_detect.*api_key")
  expect_error(bt_representation_openai(openai_model = "gpt", 
                                        documents = "test",
                                        delay_in_seconds = "test"), "is.numeric.*delay_in_seconds")
  expect_error(bt_representation_openai(openai_model = "gpt", 
                                        documents = "test",
                                        prompt = 8), "is.character.*prompt")
  expect_error(bt_representation_openai(openai_model = "gpt", 
                                        documents = "test",
                                        diversity = "test"), "is.numeric.*diversity")
  expect_error(bt_representation_openai(openai_model = "gpt", 
                                        fitted_model = "test",
                                        documents = "test"), "Model should be a BERTopic model")
  expect_error(bt_representation_openai(openai_model = "gpt", 
                                        fitted_model = unfitted_model,
                                        documents = "test"), "BERTopic model is not fitted, use bt_fit_model to fit.")
  expect_error(bt_representation_openai(fitted_model = model,
                                        openai_model = "gpt",
                                        chat = FALSE,
                                        documents = "test"), "If using a gpt model, you must specify chat = TRUE")
  
  if(interactive()){expect_error(bt_representation_openai(fitted_model = model,
                                                          documents = "test",
                                                          openai_model = "test",
                                                          api_key = Sys.getenv("OPENAI_API_KEY")
  ),
  "The input model, test, is not an available OpenAI model.")}
  
  
  # extra inputs
  # expect_error(bt_representation_openai(test_input = "test",
  #                                       api_key = Sys.getenv("OPENAI_API_KEY")),
  #              "Bad argument\\(s\\) attempted to be sent to OpenAI\\(\\): test_input")
})

if(interactive()){
  test_that("bt_representation_openai returns correct output on correct input", {
      
      bt <- reticulate::import("bertopic")
      docs <- stringr::sentences[1:10]
      embeddings <- array(runif(3840), dim = c(10,384))
      model <- bt$BERTopic(embedding_model = bt_empty_embedder(),
                           umap_model = bt_empty_reducer())$fit(docs, embeddings)
      nr_topics <- model$get_topic_info() %>% nrow()
      
      representation_openai <- bt_representation_openai(fitted_model = model,
                                                        openai_model = "gpt-4o-mini",
                                                        chat = TRUE,
                                                        documents = docs,
                                                        api_key = Sys.getenv("OPENAI_API_KEY")
      )
      
      expect_true("-1" %in% names(representation_openai))
      expect_equal(nr_topics, length(representation_openai))
      expect_true(is.character(unlist(representation_openai)))
  
    
  })
}

test_that("bt_representation_hf errors on bad input", {
  
  bt <- reticulate::import("bertopic")
  model_unfitted <- bt$BERTopic()
  
  sentences <- stringr::sentences[1:50] # docs
  embeddings <- array(runif(100), dim = c(50, 2))
  model <- bt$BERTopic(
    embedding_model = bt_empty_embedder(),
    umap_model = bt_empty_reducer(),
  )$fit(sentences, embeddings = embeddings) 
  
  # standard inputs
  expect_error(bt_representation_hf(task = 6),
               "is.character.*task")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = 6),
               "is.character.*hf_model")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = 6),
               "is.character.*documents")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    default_prompt = NULL),
               "!is.null.*default_prompt.*custom_prompt")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    default_prompt = 6),
               "is.character.*default_prompt.*is.null.*default_prompt")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    custom_prompt = 6),
               "is.character.*custom_prompt.*is.null.*custom_prompt")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    nr_samples = "test"),
               "is.numeric.*nr_samples")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    nr_repr_docs = "test"),
               "is.numeric.*nr_repr_docs")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    diversity = "test"),
               "is.numeric.*diversity")
  
  # model detection is working:
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    fitted_model = "test"),
               "Model should be a BERTopic model")
  
  expect_error(bt_representation_hf(task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    fitted_model = model_unfitted),
               regexp = "BERTopic model is not fitted, use bt_fit_model to fit.")
  
  
  # extra inputs
  expect_error(bt_representation_hf(test_input = "test",
                                    task = "text",
                                    hf_model = "text",
                                    documents = "text",
                                    fitted_model = model),
               "Bad argument\\(s\\) attempted to be sent to pipeline\\(\\): test_input")
})



test_that("bt_representation_hf returns correct output on correct input", {
  
  bt <- reticulate::import("bertopic")
  sentences <- stringr::sentences[1:150] # docs
  embeddings <- array(runif(300), dim = c(150, 2))
  model <- bt$BERTopic(
    embedding_model = bt_empty_embedder(),
    umap_model = bt_empty_reducer(),
    nr_topics = 2L
  )$fit(sentences, embeddings = embeddings) 
  # model$get_topic_info()
  nr_topics <- model$get_topic_info() %>% nrow()
  
  representation_hf <- bt_representation_hf(task = "text-generation",
                                            fitted_model = model,
                                                 hf_model = "gpt2",
                                                 documents = sentences)
  
  expect_true("-1" %in% names(representation_hf))
  expect_equal(nr_topics, length(representation_hf))
  expect_true(is.character(unlist(representation_hf)))
})

# bt <- reticulate::import("bertopic")
# sentences <- stringr::sentences[1:150] # docs
# embeddings <- array(runif(300), dim = c(150, 2))
# for (i in 1:100){
#  
#   model <- bt$BERTopic(
#     embedding_model = bt_empty_embedder(),
#     umap_model = bt_empty_reducer(),
#     nr_topics = 2L
#   )$fit(sentences, embeddings = embeddings) 
#   
#   representation_hf <- bt_representation_hf(task = "text-generation",
#                                             hf_model = "gpt2",
#                                             documents = sentences,
#                                             topic_model = model)
#   
#   test_error <- representation_hf[[2]]
#   
#   
# }

