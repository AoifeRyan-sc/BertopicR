#' Create an embedding model using [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html)
#'
#' Initially this function is built upon the `sentence_transformers` Python library, but it may be expanded to accept other frameworks. You should feed in your documents as a list. You can use hardware accelerators e.g. GPUs, to speed up computation.
#'
#' @param model Name of embedding model as a string (not case sensitive)
#'
#' @return an embedding model, formed according to the model defined, that can be input to be_do_embedding to create document embeddings
#' @export
#'
#' @examples
#' embedder <- bt_make_embedder_st("all-miniLM-L6-v2")
#'
#' embedder <- bt_make_embedder_st("aLL-minilm-l6-v2")
bt_make_embedder_st <- function(model) {
  
  #Can leave space for a second argument, which is model_source and then use switch() to allow for embedding models other than sentence_transformers if the need arises.
  
  if(!is.character(model)){
    stop("'model' should be a string of text e.g. 'all-miniLM-L6-v2")
  }
  
  #Import sentence transformers to embed documents. In the future we might want to use an argument + switch to allow the user to use other platforms for embeddings.
  sentence_transformers <- reticulate::import("sentence_transformers")
  
  #Instantiate embedding model, pass ellipsis here
  embedder <- sentence_transformers$SentenceTransformer(model_name_or_path = model)
  
  attr(embedder, "embedding_model") <- model
  
  return(embedder)
}

#' Create an embedding model using a model available from the [Spacy Library](https://spacy.io/models)
#'
#' @param model The pipeline used to make predictions
#' @param ... additional arguments to be sent to the spacy.load function
#' @param prefer_gpu if TRUE use gpu if available
#' @param exclude name of pipeline components to exclude
#'
#' @return an embedding model, formed according to the model defined, that can be input to be_do_embedding to create document embeddings
#' @export
#'
#' @examples
#' \donttest{
#' # specify a non-transformer model, excluding features not required
#' embedder <- bt_make_embedder_spacy(model = "en_core_web_md", exclude = c("tagger", "parser", "ner", "attribute_ruler", "lemmatizer"))
#' 
#' # specify a transformer model and exclude features not required
#' embedder <- bt_make_embedder_spacy(model = "en_core_web_trf", exclude = c("tagger", "parser", "ner", "attribute_ruler", "lemmatizer"))
#' }
bt_make_embedder_spacy <- function(model, ..., prefer_gpu = TRUE, exclude = NULL){
  
  # input argument validation ----
  # is spacy in installed packages?
  installed_packages <- reticulate::py_list_packages()
  
  if(!"spacy" %in% installed_packages[["package"]]){
    message("spacy is not in installed packages of current environment, run reticulate::py_install(\"spacy\").")
  } 
  
  stopifnot(is.character(model),
            is.null(exclude) | is.character(exclude) | is.list(exclude),
            is.logical(prefer_gpu))
  # end validation ----
  
  spacy <- reticulate::import("spacy")
  
  if (prefer_gpu){
    spacy$prefer_gpu() # use gpu 
  }
  
  # Try loading the requested model - sometimes the model will have to be downloaded
  if (!is.null(exclude)){ # if exclude is null, do not enter as argument as default is from spacy utils
    nlp <- try(spacy$load(name = model, exclude = exclude, ...), silent = TRUE) 
  } else{
    nlp <- try(spacy$load(name = model, ...), silent = TRUE) 
  }
  
  # if there is an error loading the model, two possible reasons are:
  # 1. bad argument keyword argument
  # 2. Model needs to be downloaded 
  if(any(class(nlp) == "try-error")){
    
    # check if bad keyword argument - py_call error only returns the first bad argument even if there are many, which is a limitation
    if(grepl(".*unexpected keyword argument.*", nlp[1])){
      bad_arg <- regmatches(nlp[1], regexec("'(\\w+)'", nlp[1]))[[1]][1] 
      bad_arg <- gsub("'", "", bad_arg)
      stop("Bad argument(s) attempted to be sent to spacy.load(): ", bad_arg)
    }
    
    # if there is no issue with the keyword args, check if model needs to be downloaded
    else if (grepl(".*Can't find model.*", nlp[1])){
      attempt_download <- try(spacy$cli$download(model), silent = TRUE)
      
      # if can't download model, send error message
      if(any(class(attempt_download) == "try-error")){
        stop("It doesn't look like ", model, " is a valid model from the spacy library")
      }
      
      # if there us no issues, run spacy$load with the now downloaded model, checking if there is any exclusions specified
      else {
        if (!is.null(exclude)){
          nlp <- try(spacy$load(name = model, exclude = exclude, ...)) 
        } else{
          nlp <- try(spacy$load(name = model, ...)) 
        }
      }
    }
  }  
  
  attr(nlp, "embedding_model") <- model
  
  return(nlp)
}

#' Create an embedding model using a model available from the [Flair Library](https://flairnlp.github.io/)
#'
#' @param model name of model used to create embeddings
#' @param ... Additional arguments to be passed to the selected Flair class
#' @param flair_class BertopicR is currently compatible with 4 Flair embedding classes: FlairEmbeddings, WordEmbeddings, TransformerWordEmbeddings and TransformerDocumentEmbeddings.
#' If you chose to perform word embeddings rather than document embeddings, bt_do_embedding will pool the word embeddings for each document to calculate a mean value.
#'
#' @return an embedding model, formed according to the model defined, that can be input to be_do_embedding to create document embeddings
#' @export
#'
#' @examples
#' \donttest{
#' # Flair Embedding, reducing chars_per_chunk to help with memory issues
#' embedder <- bt_make_embedder_flair(model = "news-forward", flair_class = "FlairEmbeddings", chars_per_chunk = 400L)
#' 
#' # Transformer Document Embedding
#' embedder <- bt_make_embedder_flair(model = "roberta-base", flair_class = "TransformerDocumentEmbeddings")
#' }
bt_make_embedder_flair <- function(model,
                                   ...,
                                   flair_class = c("FlairEmbeddings",
                                                   "TransformerWordEmbeddings",
                                                   "TransformerDocumentEmbeddings",
                                                   "WordEmbeddings")){
  
  # input argument validation ----
  # is spacy in installed packages?
  installed_packages <- reticulate::py_list_packages()
  
  if(!"flair" %in% installed_packages[["package"]]){
    stop("flair is not in installed packages of current environment, run reticulate::py_install(\"flair\").\n
            Note that if you receive a module not found error, you may need to instead run reticulate::py_install(\"flair\", pip = TRUE) to force installation with pip instead of conda.")
  } 
  
  stopifnot(is.character(model))
  flair_class <- match.arg(flair_class)
  # end of initial validation steps ----
  
  flair <- reticulate::import("flair")
  
  embedding_constructors <- c(
    FlairEmbeddings = flair$embeddings$FlairEmbeddings,
    TransformerWordEmbeddings = flair$embeddings$TransformerWordEmbeddings,
    TransformerDocumentEmbeddings = flair$embeddings$TransformerDocumentEmbeddings,
    WordEmbeddings = flair$embeddings$WordEmbeddings
  )
  
  constructor_matched <- embedding_constructors[flair_class]
  
  embedder <- try(unlist(constructor_matched, use.names = FALSE)[[1]](model, ...), silent = TRUE)
  
  # check if bad keyword argument - py_call error only returns the first bad argument even if there are many, which is a limitation
  if(any(class(embedder) == "try-error")){
    if(grepl(".*The given model.*", embedder[1])){
      stop("It doesn't look like ", model, " is a valid model from the flair library")
    } 
    # check if bad keyword argument - py_call error only returns the first bad argument even if there are many, which is a limitation
    else if (grepl(".*unexpected keyword argument.*|.*unused argument", embedder[1])){
      bad_arg <- regmatches(embedder[1], regexec("'(\\w+)'", embedder[1]))[[1]][1] 
      bad_arg <- gsub("'", "", bad_arg)
      constructor_message <- stringr::str_match(as.character(unlist(constructor_matched, use.names = FALSE)[[1]]), "'([^']+)'")[, 2]
      stop("Bad argument(s) attempted to be sent to ", constructor_message,": ", bad_arg)
    } else{
      stop(embedder)
    }
  } 
  
  attr(embedder, "embedding_model") <- model
  
  return(embedder)
}

#' Create an embedding model using a model available from the [OpenAI Library](https://platform.openai.com/docs/models/embeddings)
#'
#' @param model name of openai model used to create embeddings
#' @param openai_api_key OpenAI API key is required to use the OpenAI API and can be found on the OpenAI website
#'
#' @return an embedding model, formed according to the model defined, that can be input to be_do_embedding to create document embeddings
#' @export
#'
#' @examples
#' \dontrun{
#' embedder <- bt_make_embedder_openai(model = "text-embedding-ada-002", openai_api_key = "sk-")
#' }
bt_make_embedder_openai <- function(model = "text-embedding-ada-002",
                                    openai_api_key = "sk-"){
  # input validation ---
  stopifnot(is.character(model),
            is.character(openai_api_key))
  
  bt_backend <- reticulate::import("bertopic.backend")
  openai <- reticulate::import("openai")
  openai$api_key <- openai_api_key
  
  models <- openai$Model$list()$data %>% 
    lapply(function(sublist) sublist$id) %>% 
    unlist() # all openai models
  
  # there is currently only 1 embedding model available the adheres to this rule - future implementations might not adhere?
  openai_embedding_models <- models[grep("embedding", models, ignore.case = TRUE)]
  
  # Is the input model available?
  if (!model %in% models){
    stop("The input model, ", model, ", is not an available OpenAI embedding model.")
  }
  # end of input validation ----
  
  embedding_model <- bt_backend$OpenAIBackend(model)
  
  attr(embedding_model, "embedding_model") <- model
  
  return(embedding_model)
  
}
#' Embed your documents
#'
#' Takes a document, or list of documents, and returns a numerical embedding which can be used as features for machine learning model or for semantic similarity search. If you have pre-computed your embeddings you can skip this step. the bt_embed function is designed to be used as one step in a topic modelling pipeline.
#'
#' Initially this function is built upon the `sentence_transformers` Python library, but it may be expanded to accept other frameworks. You should feed in your documents as a list. You can use hardware accelerators e.g. GPUs, to speed up computation.
#'
#' The function currently returns an object with two additional attributes: embedding_model, n_documents, they have been appended to the embeddings for extraction at later steps in the pipeline, e.g. when merging data frames later on it's important to check how many documents we entered.
#'
#' @param embedder An embedding model (output of bt_make_embedder)
#' @param documents A character vector of the documents to be embedded, e.g. your text variable
#' @param ... Optional or additional parameters passed to SentenceTransformer's encode function, e.g. batch_size
#' @param accelerator A string containing the name of a hardware accelerator, e.g. "mps", "cuda". This is currently applied only if the embedder is a sentence transformer or from the flair library. If NULL no accelerator is used for sentence transformer or flair embeddings. GPU usage for spacy embeddings should be specified on embedder creation (bt_make_embedder_spacy)
#' @param progress_bar A logical value indicating whether a progress bar is shown in the console. This is only used if using an embedder from the sentence-transformer package
#'
#' @return An array of floating point numbers
#' @export
#' 
#' @examples
#' docs <- c("i am", "a list of", "documents", "to be embedded")
#' 
#' embedder <- bt_make_embedder_st("aLL-minilm-l6-v2")
#' 
#' embeddings <- bt_do_embedding(embedder, docs, accelerator = NULL)
#' 
bt_do_embedding <- function(embedder, documents, ..., accelerator = NULL, progress_bar = TRUE) {
  
  # update this to be compatible with compatible embedding models
  # if(!grepl("^sentence_tran",class(embedder)[[1]])){
  #   stop("This package currently only supports embedding models from the sentence transformer library, embedder should be a sentence transformers model")
  # }
  
  #Store the attributes associated with the embedder for adding the embedding_model later
  embedder_attributes <- attributes(embedder)
  
  #Stop early if conditions aren't met
  stopifnot(is.character(accelerator) | is.null(accelerator),
            is.logical(progress_bar))
  
  #Create embeddings for sentence transformer
  if(grepl("^sentence_tran",class(embedder)[[1]])){
    embeddings <-
      embedder$encode(
        documents,
        device = accelerator,
        show_progress_bar = progress_bar,
        ... #This allows knowledgeable users to include other arguments, without bloating the autofill for inexperienced users
      )
  }
  else if(grepl("^spacy",class(embedder)[[1]])){
    # print("spacy found")
    
    embeddings <- c()
    
    for (doc in 1:length(documents)){
      
      embedding <- embedder(documents[doc])
      # print("embedding initiated")
      
      if (embedding$has_vector){
        embedding = embedding$vector
        # print("embedding has vec")
      }
      else{
        # set environment so that py_eval can access local variables
        withr::with_options(c(reticulate.engine.environment = rlang::current_env()), {
          embedding = reticulate::py_eval("r.embedding._.trf_data.tensors[-1][0]")
        })
      }
      # this following part is implemented in the python code but I haven't found an applicable use case yet
      if (!is.array(embedding) && methods::hasMethod("get", embedding)) {
        embedding = embedding$get()
      }
      embeddings <- rbind(embeddings, embedding)
    }
    attributes(embeddings)$dimnames <- NULL
  }
  else if(grepl("^flair",class(embedder)[[1]])){
    flair <- reticulate::import("flair")
    
    if (!is.null(accelerator)){
      flair$device <- flair$torch$device("mps")
    }
    
    # disable fine tune to prevent CUDA OOM error (have not experienced this myself - from BERTopic python package)
    if ("fine_tune" %in% names(embedder)){
      embedder$fine_tune <- FALSE 
    }
    
    # want document embeddings, not token embeddings so using mean pooling to convert token to document embeddings
    if (grepl("TokenEmbeddings", class(embedder)[[2]])){
      embedder <- flair$embeddings$DocumentPoolEmbeddings(embedder) 
    }
    
    embeddings <- c()
    for (doc in 1:length(documents)){
      sentence <- flair$data$Sentence(documents[doc]) # convert document to flair object
      embedding_step <- embedder$embed(sentence) # embed sentence
      embedding <- sentence$embedding$cpu()$detach()$numpy() # extract embedding and tensor values and convert values to numpy array
      embeddings <- rbind(embeddings, embedding) # concatenate document embeddings
    }
    attributes(embeddings)$dimnames <- NULL # remove dimnames attribute
  }
  else if (grepl(".*openai.*",class(embedder)[[1]])){
    embeddings <- embedder$embed(documents)
  }
  else {
    stop("embedding model passed is not currently supported by this library. Embedding models should be from one of the following sources:\n
            - sentence-transformers\n
            - flair\n
            - OpenAI\n
            - spacy")
  }
  
  
  # implement this if we introduce a hugging face embedder function
  # else if(grepl("transformers.*pipelines",class(embedder)[[1]])){
  #   
  # }
  
  
  
  #Give the user a quick console nudge that the embeddings are ready
  message("\nEmbedding proccess finished")
  
  #Keep track of the number of documents that were fed into the bt_embed function, should be useful later when merging data frames and documents aren't present any more. Should we just return a data frame with the documents, and nested embeddings?
  n_documents <- length(documents)
  attr(embeddings, "n_documents") <- n_documents
  
  
  #Add the embedding_model attribute if we can (this will help later on, or when debugging for other users.)
  if("embedding_model" %in% names(embedder_attributes)){
    attr(embeddings, "embedding_model") <- embedder_attributes$embedding_model
    message(paste0(embedder_attributes$embedding_model, " added to embeddings attributes"))
  } else{
    message("No embedding_model attribute found on embedder, proceeding without adding")
  }
  
  return(embeddings)
  
}

# do we need a bt_make_embedder_hf when we have st? Should I be using pipeline or different function to make the embedder ----
#' Create an embedding model using the [hugging face library](https://huggingface.co/models)
#'
#' @param ... Additional arguments sent to transformers.pipeline()
#' @param task The task defining the pipeline to be returned, this defaults to feature extraction.
#' @param model The model used by the pipeline to make predictions
#'
#' @return a pipeline formed according to the task defined
#' @export
#'
#' @examples
#' # define task and use default model for that task
#' embedder <- bt_make_embedder_hf(task = "feature-extraction")
#' 
#' # define model and use task specified for that model
#' embedder <- bt_make_embedder_hf(model = "distilbert-base-cased")
#' 
#' # define task and model
#' embedder <- bt_make_embedder_hf(task = "feature-extraction", model = "facebook/bart-base")
#' 
#' # define 
# bt_make_embedder_hf <- function(..., task = "feature-extraction", model = NULL){
#   # input argument validation
#   if (is.null(model) & is.null(task)){
#     stop("Either model or task input argument must be specified.")
#   }
#   
#   stopifnot(is.null(task)| is.character(task),
#             is.null(model) | is.character(model))
#   
#   dots <- rlang::list2(...) # extra inputs as list
#   transformers <- reticulate::import("transformers") # import transformers library
#   inspect <- reticulate::import("inspect") # import inspect to validate extra arguments
#   empty_model <- transformers$pipeline # function to be used with no arguments
#   available_args <- inspect$getfullargspec(empty_model)$args # arguments allowed
#   
#   if(any(!names(dots) %in% available_args)){
#     bad_args <- names(dots)[!names(dots) %in% names(empty_model)] # non-applicable args
#     stop(paste("Bad argument(s) attempted to be sent to transformers.pipeline():", bad_args, sep = ' '))
#   }
#   
#   # end input validation
#   
#   pipeline <- transformers$pipeline(task = task,
#                                     model = model,
#                                     ...)
#   
#   return(pipeline)
#   
# }