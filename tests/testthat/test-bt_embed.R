#testing bt_make_embedder_st ----
test_that("bt_make_embedder_st fails when it should and succeeds otherwise",{
  expect_error(bt_make_embedder_st("all-minilm-l6-v10"),
               regexp = "Repository Not Found")
  
  expect_silent(bt_make_embedder_st("all-minilm-l6-v2"))
  
})

test_that("bt_make_embedder_st returns the appropriate object", {
  embedder <- bt_make_embedder_st("all-minilm-l6-v2")
  
  expect_true(grepl("^sentence_tran",class(embedder)[[1]]))
  expect_true("encode" %in% names(embedder))
  
  #Check we have an attribute named embedding model
  embedder_attributes <- attributes(embedder)
  expect_true("embedding_model" %in% names(embedder_attributes))
  
})

# testing bt_make_embedder_spacy ----
if(interactive()){
  test_that("bt_make_embedder_spacy fails when it should and succeeds otherwise",{

    # bad model
    expect_error(bt_make_embedder_spacy("test"),
                 regexp = "It doesn't look like test is a valid model from the spacy library")

    # bad exclude arg
    expect_error(bt_make_embedder_spacy(model = "test",
                                        exclude = 2),
                 regexp = "is.null\\(exclude\\).*")

    # bad arg
    expect_error(bt_make_embedder_spacy(model = "test",
                                        test_arg = 3),
                 regexp = "Bad argument\\(s\\) attempted to be sent to spacy.load\\(\\): test_arg")

    # bad gpu arg
    expect_error(bt_make_embedder_spacy(model = "test",
                                        prefer_gpu = 2),
                 regexp = "is.logical\\(prefer_gpu\\).*")

    expect_silent(bt_make_embedder_spacy(model = "en_core_web_sm",
                                         exclude = "tagger",
                                         vocab = FALSE))


  })
}

if(interactive()){

  test_that("bt_make_embedder_spacy returns the appropriate object", {

    embedder <- bt_make_embedder_spacy("en_core_web_sm",
                                       exclude = c("tagger", "parser", "ner", "attribute_ruler", "lemmatizer"))

    expect_true(grepl("^spacy",class(embedder)[[1]]))
    expect_true(!all(c("tagger", "parser", "ner", "attribute_ruler", "lemmatizer") %in% embedder$component_names))

    #Check we have an attribute named embedding model
    embedder_attributes <- attributes(embedder)
    expect_true("embedding_model" %in% names(embedder_attributes))

  })
}

if(interactive()){

  test_that("bt_make_embedder_flair fails when it should and succeeds otherwise",{

    # bad model
    expect_error(bt_make_embedder_flair("test"),
                 regexp = "It doesn't look like test is a valid model from the flair library")

    # bad flair_class arg
    expect_error(bt_make_embedder_flair(model = "test",
                                        flair_class = "test"),
                 regexp = "\'arg\' should be one of.*")

    # bad arg
    expect_error(bt_make_embedder_flair(model = "test",
                                        test_arg = 3),
                 regexp = "Bad argument\\(s\\) attempted to be sent to flair.embeddings.token.FlairEmbeddings:")

    expect_silent(bt_make_embedder_flair(model = "news-forward",
                                         flair_class = "FlairEmbeddings",
                                         chars_per_chunk = 500L))



  })
}
# 
#  
if(interactive()){
  test_that("bt_make_embedder_flair returns the appropriate object", {

    embedder <- bt_make_embedder_flair(model = "sentence-transformers/all-miniLM-L6-v2",
                                       flair_class = "TransformerDocumentEmbeddings",
                                       layer_mean = TRUE)

    expect_true(grepl("^flair",class(embedder)[[1]]))
    expect_true(grepl(".*TransformerDocumentEmbedding.*",class(embedder)[[1]]))
    expect_true(embedder$layer_mean)
    expect_true("embed" %in% names(embedder))

    #Check we have an attribute named embedding model
    embedder_attributes <- attributes(embedder)
    expect_true("embedding_model" %in% names(embedder_attributes))
#     
  })
}


if(interactive()){
  test_that("bt_make_embedder_openai fails when it should and succeeds otherwise",{

    # bad model
    expect_error(bt_make_embedder_openai(model = "test",
                                         openai_api_key = Sys.getenv("OPENAI_API_KEY")),
                 regexp = "The input model, test, is not an available OpenAI embedding model.")

    # bad api_key arg
    expect_error(bt_make_embedder_openai(model = "test",
                                         openai_api_key = 3),
                 regexp = "is.character\\(openai_api_key\\) is not TRUE")

    expect_silent(bt_make_embedder_openai(openai_api_key = Sys.getenv("OPENAI_API_KEY")))
  })
}

if(interactive()){
  test_that("bt_make_embedder_openai returns the appropriate object", {

    embedder <- bt_make_embedder_openai(openai_api_key = Sys.getenv("OPENAI_API_KEY"))

    expect_true(grepl(".*openai.*", class(embedder)[[1]]))
    expect_true("embed" %in% names(embedder))

    #Check we have an attribute named embedding model
    embedder_attributes <- attributes(embedder)
    expect_true("embedding_model" %in% names(embedder_attributes))


  })
}

#Testing bt_do_embedding ----

test_that("bt_do_embedding function is raising errors when it should", {
  
  embedder <- bt_make_embedder_st("all-minilm-l6-v2")
  expect_error(
    bt_do_embedding(
      embedder = 1L,
      documents = "text",
    ),
    regexp = '^embedding model passed is not currently supported by this library.*'
  )
  
  expect_error(
    bt_do_embedding(
      embedder = embedder,
      documents = "text",
      accelerator = 1),
    regexp = "is.character\\(accelerator\\) | is.null\\(accelerator\\) is not TRUE")
  
  expect_error(
    bt_do_embedding(
      embedder = embedder,
      documents = "text",
      accelerator = "mps",
      progress_bar = 1L),
    regexp = "is.logical\\(progress_bar\\) is not TRUE"
  )
  
})

test_that("bt_do_embedding function integrates with bt_make_embedder_st and returns an array when fed correct parameters and has appropriate attributes", {
  
  embedder <- bt_make_embedder_st("all-minilm-l6-v2")
  
  #Function takes one text
  test_embeddings <- bt_do_embedding(
    embedder = embedder,
    documents = "text",
    accelerator = "mps",
    progress_bar = TRUE)
  
  expect_equal(384, dim(test_embeddings))
  
  #Function takes a vector of texts
  embeddings_list <- bt_do_embedding(
    embedder = embedder,
    documents = c("this is some text", "this is more text"))
  
  expect_true(all(class(embeddings_list) == c("matrix", "array")))
  
  #Function passes ... appropriately, here we pass batch_size through ..., as batch_size is not a named argument in our function, we'll know ... is behaving as expected if we're able to pass batch_size and it then has the desired behaviour.
  ellipsis <- bt_do_embedding(
    embedder = embedder,
    documents = c("this is some text", "this is more text"),
    batch_size = 1L)
  
  expect_equal(class(ellipsis), c('matrix', "array"))
  
  #Check our ellipsis object has 3 attributes
  expect_equal(length(attributes(ellipsis)), 3)
  
  #Check n_documents attributes is as it should be (n_doc doesn't have to exact match)
  ellipsis_docs <- attr(ellipsis, "n_doc")
  expect_equal(ellipsis_docs, 2)
})

test_that("bt_do_embedding function integrates with bt_make_embedder_spacy and returns an array when fed correct parameters and has appropriate attributes", {

  embedder <- bt_make_embedder_spacy("en_core_web_sm")

  #Function takes one text
  test_embeddings <- bt_do_embedding(
    embedder = embedder,
    documents = "text")

  expect_equal(96, dim(test_embeddings)[2])
  expect_equal(1, dim(test_embeddings)[1])

  #Function takes a vector of texts
  embeddings_list <- bt_do_embedding(
    embedder = embedder,
    documents = c("this is some text", "this is more text"))

  expect_true(all(class(embeddings_list) == c("matrix", "array")))

  #Check n_documents attributes is as it should be (n_doc doesn't have to exact match)
  n_docs <- attr(embeddings_list, "n_doc")
  expect_equal(n_docs, 2)
})

if(interactive()){
  test_that("bt_do_embedding function integrates with bt_make_embedder_flair and returns an array when fed correct parameters and has appropriate attributes", {
    
    embedder <- bt_make_embedder_flair(model = "news-forward")
    
    #Function takes one text
    test_embeddings <- bt_do_embedding(
      embedder = embedder,
      documents = "text")
    
    expect_equal(2048, dim(test_embeddings)[2])
    expect_equal(1, dim(test_embeddings)[1])
    
    #Function takes a vector of texts
    embeddings_list <- bt_do_embedding(
      embedder = embedder,
      documents = c("this is some text", "this is more text"))
    
    expect_true(all(class(embeddings_list) == c("matrix", "array")))
    
    #Check n_documents attributes is as it should be (n_doc doesn't have to exact match)
    n_docs <- attr(embeddings_list, "n_doc")
    expect_equal(n_docs, 2)
  })
}

if(interactive()){
  test_that("bt_do_embedding function integrates with bt_make_embedder_openai and returns an array when fed correct parameters and has appropriate attributes", {
    
    embedder <- bt_make_embedder_openai(openai_api_key = Sys.getenv("OPENAI_API_KEY"))
    
    #Function takes one text
    test_embeddings <- bt_do_embedding(
      embedder = embedder,
      documents = "text")
    
    expect_equal(1536, dim(test_embeddings)[2])
    expect_equal(1, dim(test_embeddings)[1])
    
    #Function takes a vector of texts
    embeddings_list <- bt_do_embedding(
      embedder = embedder,
      documents = c("this is some text", "this is more text"))
    
    expect_true(all(class(embeddings_list) == c("matrix", "array")))
    
    #Check n_documents attributes is as it should be (n_doc doesn't have to exact match)
    n_docs <- attr(embeddings_list, "n_doc")
    expect_equal(n_docs, 2)
  })
}


test_that("embedding_model attribute persists when it should and doesn't break stuff when it doesn't",{
  embedder <- bt_make_embedder_st("all-minilm-l6-v2")
  expect_true("embedding_model" %in% names(attributes(embedder)))
  
  embeddings <- bt_do_embedding(embedder, "text")
  expect_true("embedding_model" %in% names(attributes(embeddings)))
  
  attr(embedder, "embedding_model") <- NULL
  expect_message(bt_do_embedding(embedder, "text"), regexp = "No embedding_mod")
})