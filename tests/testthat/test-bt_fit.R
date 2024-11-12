test_that("bt_fit accepts a bertopic model, raises an error if not",{
  bertopic <- reticulate::import('bertopic')
  documents <- stringr::sentences[1:40]
  embeddings <- array(runif(n = 240), dim = c(40, 6))

  #Returns error if NULL object fed in to model
  empty_model <- NULL
  expect_error(bt_fit_model(empty_model,documents, embeddings))

  empty_model <- bertopic$BERTopic()
  expect_message(bt_fit_model(empty_model,documents, embeddings), 
                 "The input model has been fitted to the data and updated accordingly")
})

test_that("bt_fit_model raises an error if the dimensions of documents and (embeddings or topic_labels)  don't match up, and doesn't if they do", {
  bertopic <- reticulate::import("bertopic")
  model <- bertopic$BERTopic()

  documents <- stringr::sentences[1:100]
  incorrect_embeddings <- array(runif(100), dim = c(50, 2))
  null_labels <- NULL
  topic_labels <- rep(1:10, each = 10)
  topic_labels_bad_dims <- rep(1:9, each = 10)

  #Raise an error when they don't match up.
  expect_error(bt_fit_model(model = model, documents = documents, embeddings = incorrect_embeddings), regexp = "dimensions of your documents and embeddings do not")

  correct_embeddings <- array(runif(200), dim = c(100, 2))

  #Run without raising an error
  expect_message(bt_fit_model(model = model, documents = documents, embeddings = correct_embeddings))

  #Check we error if not array or data.frame provided
  expect_error(bt_fit_model(model = model, documents = "hello",
                        embeddings = "hello"))

  expect_error(
    bt_fit_model(model = model, documents = 1, embeddings = array()),
    regexp = "is\\.character\\(documents"
  )

  expect_error(
    bt_fit_model(
      model = model,
      documents = documents,
      embeddings = NULL,
      topic_labels = topic_labels_bad_dims
      )
    )

  expect_message(
    bt_fit_model(model,
                 documents = documents,
                 embeddings = NULL,
                 topic_labels = topic_labels
                 )
    )


})

test_that("convert_to_numpy_array takes data frames or arrays as inputs", {
  test_array <- array(runif(100), c(10, 10))
  test_df <- as.data.frame(test_array)

  expect_silent(convert_to_np_array(test_array))
  expect_silent(convert_to_np_array(test_df))
})

