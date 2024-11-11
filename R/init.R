# This script is for setting up the Python environment:
# https://rstudio.github.io/reticulate/articles/package.html
.onAttach <- function(libname, pkgname) {

  bertopicr_env <- Sys.getenv("BERTOPICR_ENV")

  if(bertopicr_env == "") {
    bertopicr_env <- "BertopicR"
  }

  #Try to use miniconda to
  result <- tryCatch(
    {reticulate::use_miniconda(bertopicr_env, required = TRUE)},
    error = function(e) {e}
  )
  if ("error" %in% class(result)) {
    if(stringr::str_detect(result$message, "Minicoda is not installed")){
      stop(
        paste0( result$message, "\nInstall Miniconda with `reticulate::install_miniconda()` and try again."))
    }
    if(stringr::str_detect(result$message, "Unable to locate conda environment")){
      packageStartupMessage(paste0("\nCreating environment, ", bertopicr_env))

      reticulate::conda_create(
        envname = bertopicr_env,
        conda = paste0(reticulate::miniconda_path(), "/condabin/conda")
      )
      packageStartupMessage(paste0("\nSuccessfully created environment ", bertopicr_env))
    }
    if(!("bertopic" %in% reticulate::py_list_packages(envname = bertopicr_env)[["package"]])){
      warning("Missing Python dependencies. Run `bertopicr::install_python_dependencies()` to install.")
    }
  }

  #Get correct Python path
  python_path <- reticulate::conda_list()
  python_path <- python_path[python_path["name"] == bertopicr_env, 2]


  #Set correct Python path for reticulate
  Sys.setenv(RETICULATE_PYTHON = python_path)

  #Load the conda env
  reticulate::use_condaenv(condaenv = bertopicr_env, required = TRUE)

  invisible()
}


#' Install Python Dependencies
#'
#' @return Nothing
#' @export
#'
install_python_dependencies <- function(){
  bertopicr_env <- Sys.getenv("BERTOPICR_ENV")
  if(bertopicr_env == "") {
    bertopicr_env <- "BertopicR"
  }

  #Taken from BERTOPIC setup.py
  #https://github.com/MaartenGr/BERTopic/blob/master/setup.py
  # bertopic_0_15_0_deps <- c("bertopic==0.15.0", "numpy==1.24.3", "hdbscan==0.8.29", "umap-learn==0.5.3", "pandas==2.0.2", "scikit-learn==1.2.2", "pytorch==2.0.0","tqdm==4.65.0", "sentence-transformers==2.2.2","plotly==5.15.0", "openai==0.27.8")
  bertopic_0_15_0_deps <- c("bertopic==0.15.0", "numpy==1.24.3", "hdbscan==0.8.29", "umap-learn==0.5.3", "pandas==2.0.2", "scikit-learn==1.2.2")
                            # , "pytorch==2.0.0","tqdm==4.65.0", "sentence-transformers==2.2.2","plotly==5.15.0", "openai==0.27.8")

  reticulate::py_install(
    bertopicr_env,
    packages = bertopic_0_15_0_deps,
  )
}

#' Check that dependencies are loaded
#'
#' @return A message confirming whether or not dependencies are loaded
#' @export
#'
check_python_dependencies <- function(){
  installed_packages <- reticulate::py_list_packages()

  if("bertopic" %in% installed_packages[["package"]]){
    message("bertopic is installed, setup looks good.")
  } else {
    message("bertopic not in installed packages of current environment.\nEither load BertopicR environment or run `bertopicr::install_python_dependencies()`")
  }
}

# get the current python environment
get_current_python_environment <- function() {
  if (Sys.info()["sysname"] == "Windows") {
    reticulate::py_config()$python %>%
      stringr::str_extract(".*(?<=/BertopicR)")
  } else {
    paste0(
      "/",
      reticulate::py_config()$python %>%
        stringr::str_extract("/.*(?<=/bin/python$)") %>%
        stringr::str_remove_all("/bin/python") %>%
        stringr::str_remove("/")
    )
  }
}



import_bertopic <- function(){
  if (!"bertopic" %in% names(reticulate::py)){
    result <- tryCatch({
      reticulate::py_run_string("import bertopic")
    },
    error = function(e) {e}
    )
  }
  if("error" %in% class(result)){
    if(stringr::str_detect(result$message, "No module name")){
      env <- get_current_python_environment()

      stop(paste0("\nMissing Python Library! Run `BertopicR::install_python_dependencies()` before proceeding"))

    }
  }
}

#' Detach bertopic from the python session
#'
#' Call this when you're finished with the topic modelling process. Although, safer may be to simply save your work and then restart your R session, as the Python session is still running (and as far as I know, there's no way to safely close)
#'
#' @return Nothing
#' @export
#'
#' @usage
#' bertopic_detach()
bertopic_detach <- function(){

  #Import sys
  reticulate::py_run_string("import sys")

  #Check if bertopic is inside the Python session and delete it if it is (this follows what spacyr does, not 100% it's good practice.)
  reticulate::py_run_string('if "bertopic" in locals():\n  del bertopic')

  x <- reticulate::py_run_string("locals()")
  py_info <- reticulate::py_to_r(x)
  if(!"bertopic" %in% names(py_info)){
    stop("bertopic was not found in the Python session")
  }

  reticulate::py_run_string("del bertopic")

  y <- reticulate::py_run_string("locals()")
  py_info_updated <- reticulate::py_to_r(y)

  if(!"bertopic" %in% names(py_info)){
    stop("bertopic was detached")
  }

}

