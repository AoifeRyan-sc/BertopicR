#' Create representation model using keybert
#'
#' @description
#' This creates topic representations based on the KeyBERT algorithm. 
#' 
#' @details
#' KeyBERT is a python package that is used for extraction of key words from 
#' documents. It works by:
#' 1. Selecting representative documents (nr_repr_docs) for each topic based on the c_tf_idf cosine similarity of documents and their topics. This is achieved by sampling nr_samples documents from each topic and calculating their c_tf_idf score and choosing the top nr_repr_docs from this.
#' 2. Candidate words (nr_candidate_words) are selected for each topic based on their c_tf_idf scores for that topic
#' 3. Topic embeddings are created by averaging the embeddings for the representative documents for each topic and compared, using cosine similarity, with candidate word embeddings to give a similarity score for each word and topic
#' 4. the top_n_words with the highest cosine similarity to a topic are used to represent that topic
#' 
#' @param fitted_model Output of bt_fit_model() or another bertopic topic model. The model must have been fitted to data.
#' @param documents The documents the fitted_model has been fitted to
#' @param document_embeddings embeddings used to fit the model, these should have the same dimensions as that specified by the embedder you pass as the embedding model
#' @param embedding_model The model used to create the embeddings passed. This will be used to create word embeddings that will be compared to topic embeddings using cosine similarity.
#' @param top_n_words number of keywords/phrases to be extracted
#' @param nr_repr_docs number of documents used to create topic embeddings
#' @param nr_samples number of samples to select representative docs from for each topic
#' @param nr_candidate_words number of words to examine per topic
#'
#' @return KeyBERTInspired representation model
#' @export
#' 
#' @usage bt_representation_keybert(
#' fitted_model, 
#' documents, 
#' document_embeddings,
#' embedding_model,
#' top_n_words = 10,
#' nr_repr_docs = 50,
#' nr_samples = 500,
#' nr_candidate_words = 100)
#'
bt_representation_keybert <- function(fitted_model,
                                      documents,
                                      document_embeddings,
                                      embedding_model,
                                      top_n_words = 10L,
                                      nr_repr_docs = 50L,
                                      nr_samples = 500L,
                                      nr_candidate_words = 100L){
  
  #### input validation ####
  stopifnot(is.numeric(top_n_words),
            is.numeric(nr_repr_docs),
            is.numeric(nr_samples),
            is.numeric(nr_candidate_words),
            is.character(documents),
            is.numeric(document_embeddings), is.array(document_embeddings) | is.data.frame(document_embeddings)) # correct format
  
  if(!is.null(fitted_model)) {
    test_is_fitted_model(fitted_model) # model fitted
  }
  
  #If the embedder isn't a sentence transformers object, stop early.
  if(!grepl("^sentence_tran",class(embedding_model)[[1]])){
    stop("This package currently only supports embedding models from the sentence transformer library, embedder should be a sentence transformers model")
  }
  
  test_embeddings_dims(documents, document_embeddings)
  
  #### end validation ####
  
  # This works by:
  # 1. extract the top n representative docs per topic
  # 2. top n words per topic
  # 3. embeddings for words and representative docs (and topics by averaging representative docs per topic)
  # 4. similar words extracted by cosine similarity between word and topic embeddings
  
  # 1. first we should get the representative docs ----
  # pandas df with Topic and Document cols for input to _extract_representative_docs()
  docs_df <- data.frame(Document = documents,
                        Topic = fitted_model$get_document_info(documents)$Topic) %>% reticulate::r_to_py()
  
  c_tf_idf <- fitted_model$c_tf_idf_ # ctfidf for input to _extract_representative_docs
  docs_ind <- list()  # initiate list variable for multiple outputs from _extract_representative_docs
  
  nr_samples <- as.integer(nr_samples) # integer
  nr_repr_docs <- as.integer(nr_repr_docs) # integer
  
  withr::with_options(c(reticulate.engine.environment = rlang::current_env()), {
    docs_ind <- reticulate::py_eval("r.fitted_model._extract_representative_docs(r.c_tf_idf, r.docs_df, r.fitted_model.topic_representations_, r.nr_samples, r.nr_repr_docs)")
  })
  # assign correct list components
  representative_docs <- docs_ind[[2]]
  repr_doc_indices <- docs_ind[[4]]
   
  # 2. now we extract the candidate words from each topic ----
  
  # import necessary modules
  bt <- reticulate::import("bertopic")
  np <- reticulate::import("numpy")
  
  words <- fitted_model$vectorizer_model$get_feature_names_out() # get all words used in vectoriser out 
  nr_candidate_words <- as.integer(nr_candidate_words) # convert to int
  
  # function variables are not visible to python virtual environment, must force current environment
  withr::with_options(c(reticulate.engine.environment = rlang::current_env()), {
    # extract indices of top n values in c_tf_idf score per word per topic matrix - this indexing will start at 0
    indices <- reticulate::py_eval("r.fitted_model._top_n_idx_sparse(r.c_tf_idf, r.nr_candidate_words)", convert = FALSE)
    
    # get the scores of the top n values in c_tf_idf score per word per topic matrix
    scores <- reticulate::py_eval("r.fitted_model._top_n_values_sparse(r.c_tf_idf, r.indices)")
  })
  
  # indices of scores required to sort the scores from smallest to largest - this indexing will start at 0
  # sorted_indices <- reticulate::py_eval("r.np.argsort(r.scores, 1)") %>%
  #   np$asarray(dtype = np$int16) %>% reticulate::r_to_py()
  sorted_score_indices <- np$argsort(scores, 1L) %>% 
    np$asarray(dtype = np$int32) %>%
    reticulate::r_to_py()
  
  # function variables are not visible to python virtual environment, must force current environment
  withr::with_options(c(reticulate.engine.environment = rlang::current_env()), {
    # sort word indices and word scores based on score value - indices will start at 0
    indices_sorted <- reticulate::py_eval("r.np.take_along_axis(r.indices, r.sorted_score_indices, axis=1)", convert = FALSE)
    scores_sorted <- reticulate::py_eval("r.np.take_along_axis(r.scores, r.sorted_score_indices, axis=1)", convert = FALSE) 
  })
  
  # Create an empty list to store the topic words
  topics_list <- list()
  topic_embedding <- list()
  topics <- fitted_model$topic_representations_
  
  for (index in seq_along(names(topics))) {
    label <- names(topics)[index]
    
    # Reverse the indices and scores for to be decreasing in order of size
    py_index <- index - 1 # python indexing starts at 0
    reversed_indices <- indices_sorted[py_index] %>% reticulate::py_to_r() %>% rev() # need to convert back to r format
    # reversed_indices_r_indexing <- reversed_indices + 1L
    reversed_scores <- scores_sorted[py_index] %>% reticulate::py_to_r() %>% rev()
    
    # Create an empty list to store info on this topic
    topic_info <- list()
    
    for (i in seq_along(reversed_indices)) {
      word_index <- reversed_indices[[i]] 
      score <- reversed_scores[[i]]
      
      if (!is.null(word_index) & score > 0) {
        topic_info[[length(topic_info) + 1]] <- words[word_index + 1L] # indexing in reversed_indices starts at 0, need to adjust for r indexing starting at 1
      } else {
        topic_info[[length(topic_info) + 1]] <- NULL
      }
    }
    
    # add current topic info to the final topic list
    topics_list[[label]] <- topic_info
  }
  
  # 3. now compare word and topic embeddings ----
  repr_doc_embedding <- list()
  topic_embeddings <- list()
  
  # get topic embeddings
  for (topic in seq_along(repr_doc_indices)){
    repr_doc_embedding <- document_embeddings[repr_doc_indices[[topic]],] 
    topic_embeddings[[topic]] <- colMeans(repr_doc_embedding)
  }
  
  # 4. extract the representations ----
  vocab <- unlist(sapply(topics_list, "[")) %>% unique()
  word_embeddings <- embedding_model$encode(vocab) %>% np$asarray()
  cosine_sim <- reticulate::import("sklearn.metrics.pairwise")
  similarity_matrix <- cosine_sim$cosine_similarity(topic_embeddings, word_embeddings)
  
  # first get the indices of each candidate word in the vocab list for each topic 
  indices_list <- lapply(topics_list, function(topic_words) { # iterate over sublists
    sapply(topic_words, function(word) { # iterate over words in sublist
      which(vocab == word) # get index of words in vocab vector
    })
  })
  
  # now that we have the indices we can extract the similarity score for each word and their corresponding topic 
  # and output a list with scores for each topic in a sublist
  updated_representation <- list()
  for (i in seq_along(indices_list)){
    join_data <- data.frame(words = vocab[indices_list[[i]]],
                            score = similarity_matrix[i, indices_list[[i]]]) %>%
      dplyr::arrange(dplyr::desc(score)) %>%
      # filter(!stringr::str_detect(words, "")) %>%
      dplyr::filter(!is.null(words)) %>%
      utils::head(top_n_words)
      
    updated_representation[[i]] <- join_data %>% dplyr::pull(words) %>%
      paste(collapse = "_")
  }
 
  names(updated_representation) <- names(topics)
  if (names(updated_representation)[1] == "-1"){
    updated_representation[[1]] = "outliers"
  }
  return(updated_representation)

}

#' Create representation model using Maximal Marginal Relevance 
#' 
#' @description
#' Calculates the maximal marginal relevance between candidate words and documents.
#' Considers similarity between keywords and phrases and already selected keywords 
#' and phrases and chooses representation based on this to maximise diversity.
#' 
#'
#' @param fitted_model Output of bt_fit_model() or another bertopic topic model. The model must have been fitted to data.
#' @param embedding_model embedding model used to embed keywords selected as potential representative words. Only compatible with sentence transformer models at this point.
#' @param diversity  How diverse representation words/phrases are. 0 = not diverse, 1 = completely diverse
#' @param top_n_words Number of keywords/phrases to be extracted
#'
#' @return MaximalMarginalRelevance representation model
#' @export
#'
#' @usage bt_representation_mmr(
#' fitted_model,
#' embedding_model,
#' diversity = 0.1,
#' top_n_words = 10)
bt_representation_mmr <- function(fitted_model,
                                  embedding_model,
                                  diversity = 0.1,
                                  top_n_words = 10L){
  
  #### input validation ####
  
  stopifnot(is.numeric(diversity), diversity >= 0, diversity <= 1,
            is.numeric(top_n_words))
  
  if(!is.null(fitted_model)) {
    test_is_fitted_model(fitted_model) # model fitted
  }
  
  #If the embedder isn't a sentence transformers object, stop early.
  if(!grepl("^sentence_tran",class(embedding_model)[[1]])){
    stop("This package currently only supports embedding models from the sentence transformer library, embedder should be a sentence transformers model")
  }
  #### end validation ####
  
  updated_topics <- list()
  topics <- fitted_model$topic_representations_
  np <- reticulate::import("numpy")
  sk_pair <- reticulate::import("sklearn.metrics.pairwise")
  
  for (topic in seq_along(topics)){
    
    # extract the words to embed
    topic_words_list <- topics[[topic]]
    topic_words <- sapply(topic_words_list, "[", 1)
    topic_words <- unlist(topic_words)
    
    # embed the words
    word_embeddings <- embedding_model$encode(topic_words)
    
    # topic embeddings - don't really like how this is done?
    topic_words_join <- paste(topic_words, collapse = " ")
    topic_embeddings <- embedding_model$encode(topic_words_join)
    
    # reshape so that cosine_similarity function accepts it as input
    topic_embeddings_reshaped <- matrix(topic_embeddings, nrow = 1, ncol = length(topic_embeddings))
    
    # cosine similarity between words per topic and between individual words and overall words per topic:
    # this is to score words based on their similarity to the topic and their similarity to other words that could represent the topic
    word_topic_sim <- sk_pair$cosine_similarity(word_embeddings, topic_embeddings_reshaped)
    intra_word_sim <- sk_pair$cosine_similarity(word_embeddings)
    
    # chose best keywords
    keywords_idx <- which.max(word_topic_sim) # this is the word with the highest similarity to the topic
    candidates_idx <- unlist(sapply(seq_along(topic_words), function(i) if (i != keywords_idx[1]) i)) # this is the index of all remaining words
    
    for (i in 0:(top_n_words-1)){
      # extract similarities within candidates and between candidates and keywords
      word_topic_similarities <- word_topic_sim[candidates_idx] # this is the scores of all remaining words when compared with the overall group of words for that topic
      word_word_similarities <- intra_word_sim[candidates_idx] # this is the scores of all if the remaining words when compared to eachother
      
      # calculate mmr
      mmr <- (1-diversity) * word_topic_similarities - diversity * word_word_similarities # calculates mmr
      mmr_idx <- candidates_idx[which.max(mmr)] # index of max mmr score and maps that to the keyword index
      
      # Update keywords & candidates
      keywords_idx <- c(keywords_idx, mmr_idx) # adds index of next representative word to keywords_idx
      candidates_idx <- candidates_idx[candidates_idx != mmr_idx] # removes the word added to keywords from candidates
      
      # print(candidates_idx)
    }
    
    selected_words <- topic_words[keywords_idx] # reorder the topic_words in order of selection by mmr
    updated_topics[[topic]] <- paste(selected_words, collapse = "_") # join for representation
    
  }

  names(updated_topics) <- names(fitted_model$topic_representations_)  # name topic representations
  
  if (names(updated_topics)[1] == "-1"){
    updated_topics[[1]] = "outliers"
  }
  return(updated_topics)
  
}

#' Create representation model that uses OpenAI text generation models
#' 
#' @description
#' Representative documents are chosen from each topic by sampling (nr_samples)
#' a number of documents from the topic and calculating which of those documents are
#' most representative of the topic by c-tf-idf cosine similarity between the topic and the
#' individual documents. From this the most representative documents (the number
#' is defined by the nr_repr_docs parameter) is extracted and passed to  the OpenAI API 
#' to generate topic labels based on one of their Completion (chat = FALSE) or ChatCompletion 
#' (chat = TRUE) models.
#' 
#'
#' @param fitted_model Output of bt_fit_model() or another bertopic topic model. The model must have been fitted to data.
#' @param documents documents used to fit the fitted_model
#' @param openai_model openai model to use. If using a chat model, set chat = TRUE
#' @param nr_repr_docs number of representative documents per topic to send to the openai model
#' @param nr_samples Number of sample documents from which the representative docs are chosen
#' @param chat set to TRUE if using gpt-3.5 model
#' @param api_key OpenAI API key is required to use the OpenAI API and can be found on the OpenAI website
#' @param delay_in_seconds The delay in seconds between consecutive prompts, this is to avoid rate limit errors.
#' @param prompt The prompt to be used with the openai model. If NULL, the default prompt is used.
#' @param diversity diversity of documents to be sent to the huggingface model. 0 = no diversity, 1 = max diversity. 
#'
#' @return OpenAI representation model
#' @export
#' 
#' @usage bt_representation_openai(
#' fitted_model,
#' documents,
#' openai_model = "gpt-4o-mini",
#' nr_repr_docs = 10,
#' nr_samples = 500,
#' chat = TRUE,
#' api_key = "sk-",
#' delay_in_seconds = NULL,
#' prompt = NULL,
#' diversity = NULL)
#'
bt_representation_openai <- function(fitted_model,
                                     documents,
                                     openai_model ,
                                     nr_repr_docs = 10L,
                                     nr_samples = 500L,
                                     chat = FALSE,
                                     api_key = "sk-",
                                     delay_in_seconds = NULL,
                                     prompt = NULL,
                                     diversity = NULL){
  
  #### input validation ####
  
  stopifnot(is.character(documents),
            is.character(openai_model),
            is.logical(chat),
            is.numeric(nr_repr_docs),
            stringr::str_detect(api_key, "^sk-"),
            # is.logical(exponential_backoff),
            is.numeric(delay_in_seconds) | is.null(delay_in_seconds),
            is.character(prompt) | is.null(prompt),
            is.numeric(diversity) | is.null(diversity))

  if(!is.null(fitted_model)) {
    test_is_fitted_model(fitted_model) # model fitted
  }
 
  # if using gpt model, must specify chat = TRUE
  if (chat == FALSE & stringr::str_detect(openai_model, "^gpt")){
    stop("If using a gpt model, you must specify chat = TRUE")
  }
  
  # get list of available openai models
  openai <- reticulate::import("openai")
  openai$api_key <- api_key
  openai_models <- openai$Model$list()$data %>% 
    lapply(function(sublist) sublist$id) %>% 
    unlist()
  
  # Is the input model available?
  if (!openai_model %in% openai_models){
    stop("The input model, ", openai_model, ", is not an available OpenAI model.")
  }
  
  #### end of input validation ####

  # Get representative docs ----
  c_tf_idf <- fitted_model$c_tf_idf_ # ctfidf for input to _extract_representative_docs
  
  # _extract_representative_docs requires df with Document and Topic columns
  docs_df <- data.frame(Document = documents,
                        Topic = fitted_model$get_document_info(documents)$Topic) %>% reticulate::r_to_py() # pandas df with Topic and Document cols
  
  topic_representations <- fitted_model$topic_representations_ # for input to function
  
  # convert relevant numbers to integers
  nr_samples <- as.integer(nr_samples)
  nr_repr_docs <- as.integer(nr_repr_docs)
  
  # function variables are not visible to python virtual environment, must force current environment
  withr::with_options(c(reticulate.engine.environment = rlang::current_env()), {
    repr_doc_mapping <- reticulate::py_eval("r.fitted_model._extract_representative_docs(r.c_tf_idf, r.docs_df, r.fitted_model.topic_representations_,  r.nr_samples, r.nr_repr_docs, r.diversity)")[[1]]
  })
  
  
  if (!is.null(prompt)){
    prompt <- prompt # if user inputs custom prompt
  } else if (is.null(prompt) & chat){ # if chat is TRUE use chat based prompt
    prompt <- "
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"
  } else { # if chat is false, use suitable prompt
    prompt <- "
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]
Keywords: [KEYWORDS]
Topic name:"
    }
  
  
  updated_representation <- list()
  updated_prompt <- list()
  for (topic in seq_along(repr_doc_mapping)){
    
    topic_keywords <- unlist(sapply(fitted_model$topic_representations_[[topic]], "[", 1)) # extract keywords for topic
    updated_prompt <- update_prompt(prompt,
                                    repr_doc_mapping[[topic]],
                                    topic_keywords,
                                    fitted_model)
    
    updated_representation[[topic]] <- openai_api_call(updated_prompt,
                                                       delay_in_seconds,
                                                       chat,
                                                       openai_model,
                                                       api_key)
  } 
  
  names(updated_representation) <- names(repr_doc_mapping)
  
  if (names(updated_representation)[1] == "-1"){
    updated_representation[[1]] = "outliers"
  }
  return(updated_representation)
  
}




#' Use Huggingface models to create topic representation
#'
#' @details
#' Representative documents are chosen from each topic by sampling (nr_samples)
#' a number of documents from the topic and calculating which of those documents are
#' most representative of the topic by c-tf-idf cosine similarity between the topic and the
#' individual documents. From this the most representative documents (the number
#' is defined by the nr_repr_docs parameter) is extracted and passed to the huggingface
#' model and topic description predicted. 
#'
#'
#' @param fitted_model The fitted bertopic model
#' @param documents the documents the topic model was fitted to
#' @param task Task defining the pipeline that will be returned. See https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html for more information. Use "text-generation" for gpt-like models and "text2text-generation" for T5-like models
#' @param hf_model The model that will be used by the pipeline to make predictions 
#' @param ... arguments sent to the transformers.pipeline function
#' @param default_prompt Whether to use the "keywords" or "documents" default prompt. Passing a custom_prompt will render this argument NULL. Default is "keywords" prompt.
#' @param nr_samples Number of sample documents from which the representative docs are chosen
#' @param nr_repr_docs Number of representative documents to be sent to the huggingface model
#' @param diversity diversity of documents to be sent to the huggingface model. 0 = no diversity, 1 = max diversity. 
#' @param custom_prompt The custom prompt to be used in the pipeline. If not specified, the "keywords" or "documents" default_prompt will be used. Use "\[KEYWORDS\]" and "\[DOCUMENTS\]" in the prompt to decide where the keywords and documents are inserted.
#'
#' @return updated representation of each topic
#' @export
#'
#' @usage bt_representation_hf(
#' fitted_model,
#' documents,
#' task,
#' hf_model,
#' ...,
#' default_prompt = "keywords",
#' nr_samples = 500,
#' nr_repr_docs = 20,
#' diversity = 10,
#' custom_prompt = NULL
#' )
bt_representation_hf <- function(fitted_model,
                                 documents,
                                 task,
                                 hf_model,
                                 ...,
                                 default_prompt = "keywords",
                                 nr_samples = 500L,
                                 nr_repr_docs = 20L,
                                 diversity = 10L,
                                 custom_prompt = NULL){
  
  #### Validation ####
  stopifnot(!is.null(default_prompt) | !is.null(custom_prompt),
            is.character(task),
            is.character(hf_model),
            is.character(documents),
            is.character(default_prompt) | is.null(default_prompt),
            is.character(custom_prompt) | is.null(custom_prompt),
            is.numeric(nr_samples),
            is.numeric(nr_repr_docs),
            is.numeric(diversity))
  
  # Check fitted model
  if(!is.null(fitted_model)) {
    test_is_fitted_model(fitted_model)
  }
  
  # check extra arguments passed
  transformer <- reticulate::import("transformers") # import transformers library
  pipeline_args <- args(transformer$pipeline) %>% as.list() %>% names()
  
  #Convert the `...` (dot-dot-dot or ellipsis) to list for checking purposes
  dots <- rlang::list2(...)
  
  #Stop function early if bad arguments fed with ellipsis and send message to user pointing out which arguments were bad
  if(any(!names(dots) %in% pipeline_args)){
    
    bad_args <- names(dots)[!names(dots) %in% pipeline_args]
    stop(paste("Bad argument(s) attempted to be sent to pipeline():", bad_args, sep = ' '))
  }
  
  #### end of validation ####
  
  # Set prompt to use
  if (!is.null(custom_prompt)){
    prompt <- custom_prompt
  }
  else if (default_prompt == "keywords"){
    prompt <- "I have a topic described by the following keywords: [KEYWORDS]. The name of this topic is:"
  }
  else if (default_prompt == "documents"){
    prompt <- "I have a topic described by the following documents: [DOCUMENTS]. The name of this topic is:"
  }
  
  # create pipeline
  generator <- transformer$pipeline(task = task, model = hf_model, ...) # create pipeline
  
  empty_string <- ""
  updated_representation <- list()
  # record_prompt <- list()
  for (doc in seq_along(fitted_model$get_topic_info()$Topic)) {
    
    updated_prompt <- prompt # don't want to overwrite prompt
    
    if (stringr::str_detect(prompt, "\\[KEYWORDS\\]")){
      
      # representative keywords
      topic_representation <- fitted_model$topic_representations_
      
      # concatenate keywords into single string per topic
      keywords <- list()
      # words_per_topic <- list()
      for (topic in seq_along(topic_representation)){
        topic_list <- topic_representation[[topic]]
        # words_per_topic <- character()
        words_per_topic <- NULL
        for (word_info in seq_along(topic_list)){
          word_list <- topic_list[[word_info]][[1]]
          words_per_topic <- c(words_per_topic, word_list)
          keywords[[topic]] <- paste(words_per_topic, collapse = ", ")
          
        }
      }
      
      updated_prompt <- gsub("\\[KEYWORDS\\]", keywords[[doc]], updated_prompt)
    }
    
    if (stringr::str_detect(prompt, "\\[DOCUMENTS\\]")){
      
      # first get representative docs and keywords
      c_tf_idf <- reticulate::py_eval("r.fitted_model.c_tf_idf_") # ctfidf for input to _extract_representative_docs
      
      # _extract_representative_docs requires df with Document and Topic columns
      docs <- data.frame(Document = documents,
                         Topic = fitted_model$get_document_info(documents)$Topic) %>% reticulate::r_to_py() # pandas df with Topic and Document cols
      
      # convert to integers
      nr_samples <- as.integer(nr_samples)
      nr_repr_docs <- as.integer(nr_repr_docs)
      diversity = as.integer(diversity)
      
      # representative docs
      # set environment so that py_eval can access local variables
      withr::with_options(c(reticulate.engine.environment = rlang::current_env()), {
        
        # get representative docs
        representative_docs <- reticulate::py_eval("r.fitted_model._extract_representative_docs(r.c_tf_idf, r.docs, r.fitted_model.topic_representations_, r.nr_samples, r.nr_repr_docs, r.diversity)")[[1]]
      })
      format_docs <- paste(empty_string, paste0("- ", substr(representative_docs[[doc]], 1, 255)), "\n")
      join_docs <- paste(format_docs, collapse = "")
      
      updated_prompt <- gsub("\\[DOCUMENTS\\]", join_docs, updated_prompt)
    }
    
    updated_representation[[doc]] <- generator(updated_prompt)[[1]][[1]] 
    # record_prompt[[doc]] <- updated_prompt
    
  }
  names(updated_representation) <- names(fitted_model$topic_representations_) 
  
  if (names(updated_representation)[1] == "-1"){
    updated_representation[[1]] = "outliers"
  }
  return(updated_representation)
  
  # return(list("updated_representation" = updated_representation, "prompts" = record_prompt))
  
}


