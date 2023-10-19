#From https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#diversify-topic-representation
#"Since MMR is using word embeddings to diversify the topic representations, it is necessary to pass the embedding model to BERTopic if you are using pre-computed embeddings:"

#' Build a BERTopic model
#'
#' Keep `*_model` = NULL to proceed with a model made from default parameters (see each individual `make_*` function for parameters). However, it is not advisable to accept default parameters for each model; you should tune each model according to your dataset and the business question you are answering.
#'
#' @param ... Additional arguments sent to bertopic.BERTopic()
#' @param embedding_model Model for creating embeddings (Python object)
#' @param reduction_model Model for reducing embeddings' dimensions (Python object)
#' @param clustering_model Model for clustering (Python object)
#' @param vectoriser_model Model for vectorising input for topic representations (Python object)
#' @param ctfidf_model Model for performing class-based tf-idf (ctf-idf) (Python object)
#'
#' @return a BERTopic model
#' @export
#' 
#' @examples
#' \dontrun{
#' # model using all default parameters
#' model <- bt_compile_model()
#' 
#' # model with modular components already generated
#' # define embedding and reduction modules and pass to bt_compile_model
#' embedder <- bt_make_embedder_st("all-mpnet-base-v2") 
#' reducer <- bt_make_reducer_umap(n_components = 10L, n_neighbours = 20L)
#' model <- bt_compile_model(embedding_model = embedder, reduction_model = reducer)
#'
#' # Perform document embedding and reduction external to bertopic model and pass empty models to bt_compile_model
#' embedder <- bt_make_embedder_st("all-mpnet-base-v2") # embedder
#' embeddings <- bt_do_embedding(embedder, docs, accelerator = NULL) # embeddings
#' reducer <- bt_make_reducer_umap(n_components = 10L, n_neighbours = 20L) # reducer
#' reduced_embeddings <- bt_do_reducing(reducer, embeddings) # reduced embeddings
#' 
#' # skip embedding and reduction step by passing empty models
#' model <- bt_compile_model(embedding_model = bt_empty_embedder(), reduction_model = bt_empty_reducer()) 
#' 
#' }
#'
bt_compile_model <- function(..., embedding_model = NULL, reduction_model = NULL, clustering_model = NULL, vectoriser_model = NULL, ctfidf_model = NULL){

  #Check if the inputted arguments are python objects or NULL (in which case we'll make some). This doesn't check they're the right python objects though, will maybe include when full implementation is done and we know all of the embedding models, reduction models etc. and their classes to test with
    stopifnot(test_is_python_object(embedding_model) | is.null(embedding_model),
              test_is_python_object(reduction_model) | is.null(reduction_model),
              test_is_python_object(clustering_model) | is.null(clustering_model),
              test_is_python_object(vectoriser_model) | is.null(vectoriser_model),
              test_is_python_object(ctfidf_model) | is.null(ctfidf_model))
  
  #Import bertopic inside function scope
  bertopic <- reticulate::import("bertopic")

  #Convert the `...` (dot-dot-dot or ellipsis) to list for checking purposes
  dots <- rlang::list2(...)
  
  #Instantiate empty model to get valid arguments
  empty_model <- bertopic$BERTopic()
  
  #Stop function early if bad arguments fed with ellipsis and send message to user pointing out which arguments were bad
  if(any(!names(dots) %in% names(empty_model))){
    
    bad_args <- names(dots)[!names(dots) %in% names(empty_model)]
    stop(paste("Bad argument(s) attempted to be sent to BERTopic():", bad_args, sep = ' '))
  }

  #Provide a default embedding model for: Since MMR is using word embeddings to diversify the topic representations, it is necessary to pass the embedding model to BERTopic if you are using pre-computed embeddings:"
  if(is.null(embedding_model)){
    embedding_model <- bt_make_embedder_st(model = "all-mpnet-base-v2")
    message("\nNo embedding model provided, defaulting to 'all-mpnet-base-v2' model as embedder.")
    }

  #If no UMAP model given, provide empty
  if(is.null(reduction_model)){
    reduction_model <- bt_make_reducer_umap()
    message("\nNo reduction_model provided, using default 'bt_reducer_umap' parameters.")
  }

  if(is.null(clustering_model)){
    clustering_model <- bt_make_clusterer_hdbscan()
    message("\nNo clustering model provided, using hdbscan with default parameters.")
  }

  if(is.null(vectoriser_model)){
    vectoriser_model <- bt_make_vectoriser()
    message("\nNo vectorising model provided, creating model with default parameters")
  }

  if(is.null(ctfidf_model)){
    ctfidf_model <- bt_make_ctfidf()
    message("\nNo ctfidf model provided, creating model with default parameters")
  }

    #Instantiate a BERTopic object with given models
  model <- bertopic$BERTopic(
    embedding_model = embedding_model,
    umap_model = reduction_model,
    hdbscan_model = clustering_model,
    vectorizer_model = vectoriser_model,
    ctfidf_model = ctfidf_model,
    ...)

  message("\nModel built & input model updated accordingly")

  return(model)
}

#' Fit a topic model on your documents & embeddings
#'
#' @description
#' 
#' If you've already performed dimensionality reduction on your embeddings, you can feed in the reduced dimension embeddings to the embeddings argument, make sure to supply `bt_compile_model` with a base reducer (the output of `bt_base_reducer()`)
#' 
#' NOTE: The bertopic model you are working with is a pointer to a python object 
#' at a point in memory. This means that the input and the output model cannot be 
#' differentiated between without explicitly saving the model before performing 
#' this operation. We do not need to specify an output to the bt_fit_model function 
#' as the function changes the input model in place. If you do decide to explicitly assign a function output,
#' be aware that the output model and the input model will be the same as one another.
#'
#' @param model Output of bt_compile_model() or another bertopic topic model
#' @param documents Your documents to topic model on
#' @param embeddings Your embeddings, can be reduced dimensionality or not. If no embeddings provided, embedding model used reduction_model in bt_compile_model is used to calculate and reduce dimensionality of embeddings.
#' @param topic_labels Pre-existing labels, for supervised topic modelling
#'
#' @return a fitted BERTopic model
#' @export
#'
#' @examples
#' \dontrun{
#' # create the model with default parameters, then fit the model to the data
#' model <- bt_compile_model()
#' bt_fit_model(model = model, documents = docs)
#' 
#' # create the model with document embeddings already created and reduced
#' # embeddings
#' embedder <- bt_make_embedder_st(all-minilm-l6-v2)
#' embeddings <- bt_do_embedding(embedder, docs)
#' 
#' # reduced embeddings
#' reducer <- bt_make_reducer_umap()
#' reduced_embeddings <-  bt_do_reducing(reducer, embeddings)
#' 
#' # model
#' model <- bt_compile_model(embedding_model = bt_empty_embedder, reducer = bt_empty_reducer)
#' bt_fit_model(model = model, documents = docs, embeddings = reduced_embeddings)
#' 
#' }
#' 
bt_fit_model <- function(model, documents, embeddings = NULL, topic_labels = NULL){

  #Input validation
  stopifnot(
    grepl("^bertopic", methods::S3Class(model)[[1]]), #should we flick to any()?
    is.array(embeddings)| is.data.frame(embeddings) |is.null(embeddings),
    is.vector(topic_labels) | is.null(topic_labels ),
    is.character(documents)
  )

  if(!is.null(embeddings)) {
    #Check the length of documents is equal to the number of embeddings, and if not, stop.
    test_embeddings_dims(documents, embeddings)

    embeddings <- convert_to_np_array(embeddings)
  }


  if(!is.null(topic_labels)){
    #Check the length of documents is equal to the number of embeddings, and if not, stop.
    test_labels_lengths(documents, topic_labels)
  }

  fitted_model <- model$fit(documents = documents, embeddings = embeddings, y = topic_labels)

  message("\nModel is fitted")
}
