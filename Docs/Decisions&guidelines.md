My decision: build a traffic-flow forecasting and congestion alert system
I recommend selecting L2 — Smart Traffic Management System for Urban Congestion, from your problem-statement PDF. We will narrow the broad problem into a small, measurable deep-learning project that Antigravity can build and test with minimal intervention from you. 
problem statements.pdf

Recommended project
FlowCast
Deep Learning-Based Short-Horizon Traffic Flow Forecasting and Congestion Risk Classification
The system will use historical traffic observations and contextual features to predict traffic volume 15 minutes ahead, then translate the prediction into a low, moderate, or high congestion category.

Domain
Time-series forecasting

Dataset
2,101 observations

Deep-learning model
GRU neural network

User interface
Streamlit



1. Why I selected this problem
The biggest constraint is not whether Antigravity can generate Python code. It is whether we can finish a complete, defensible academic project without spending days collecting data, labelling examples, configuring external services, or debugging an oversized model.
Your assignment requires a narrow problem, an open-source dataset, EDA, a conventional baseline, a deep-learning model, training curves, evaluation metrics, saved model weights and a Streamlit or Gradio interface. It also specifies a 30-page report. 
DOC-20261012-WA0012..pdf
 DOC-20261012-WA0012..pdf

This project has a practical advantage: the dataset already exists and can be retrieved programmatically.

No manual data labelling
The UCI Traffic Flow Forecasting dataset contains 2,101 observations, with historical traffic readings and contextual features. Its stated prediction task is to forecast traffic volume 15 minutes into the future. The repository reports no missing values. 


UCI Machine Learning Repository







Automatable data pipeline
The dataset is available through the ucimlrepo Python package. Antigravity can create the download, validation, preprocessing and training scripts without requiring a Kaggle API key or a separate data-collection service. 


UCI Machine Learning Repository







Measurable results
We can compare a Random Forest baseline with a GRU model, generate training and validation curves, evaluate forecasting errors, and display congestion classification metrics. This gives us concrete experimental results for your report and viva.




No paid infrastructure required
We can train through a free Google Colab GPU runtime and run the finished Streamlit application locally. There is no need for live traffic API subscriptions, databases, authentication systems, or paid hosting for the academic prototype.



Important limitation: the UCI dataset represents 36 sensor locations along two highways in Northern Virginia and the Washington, D.C. region, not Indian roads. Our report must disclose that limitation rather than claim that the model has been validated on Indian city traffic. 


UCI Machine Learning Repository



Dataset: UCI Traffic Flow Forecasting
2. Why I rejected the other candidates
Candidate	Reason I would not choose it first
Plant disease detection	Your sample report already implements a plant-disease CNN. It is too close to an existing project to be a sensible choice under your guidelines.  SAMPLE REPORT_DEEP LEARNING MINI PROJECT STATEMENTS.pdf
Crop yield prediction or crop recommendation	Public datasets exist, but these are familiar tutorial-style projects. A generic implementation would make the uniqueness requirement harder to satisfy.
E-consultation sentiment analysis	It is a feasible NLP project, but existing public implementations already target this exact problem, and finding suitable labelled consultation comments adds uncertainty.  GitHub +1
eDNA taxonomy and biodiversity	More specialised, but sequence processing and biological validation introduce unnecessary risk for a short academic deadline.
Your assignment explicitly prohibits common/tutorial-based projects and requires original implementation. 
DOC-20261012-WA0012..pdf

Traffic forecasting is not automatically unique just because we rename it. Our differentiation will come from a clearly scoped forecasting task, a proper experimental comparison, leakage-aware preprocessing where the data permits it, congestion-risk analysis, and a working interface. You must still confirm that your faculty has not assigned this topic to another team and that it does not duplicate a previous project. I cannot verify your class's internal topic allocation sheet.
3. Technical decisions I am fixing
These are the decisions I do not want Antigravity making independently.

Dataset and preprocessing
Use the UCI dataset and preserve the original source citation. The pipeline must inspect feature names, target shape, data types, missing values and feature distributions before training. Fit all preprocessing transformations using training data only.




Models
Use Random Forest regression as the required conventional baseline and a small GRU as the main deep-learning model. Add Linear Regression as an optional simple reference. Avoid oversized networks and unnecessary pretrained models.




Evaluation
For continuous traffic-volume predictions, report MAE, RMSE and \(R^2\). In addition, derive low, moderate and high traffic categories using documented thresholds calculated from training data, then evaluate those categories using precision, recall, macro-F1 and a confusion matrix. This addresses both forecasting quality and the classification metrics in your rubric.




User interface
Build a Streamlit dashboard with a ready-to-run demonstration, optional CSV upload, next-15-minute traffic-volume prediction, congestion category, model-comparison metrics and plots. Do not require users to manually populate dozens of raw features just to test the demo.




Deliverables
Keep the project modular, version-controlled and reproducible. Include training scripts, tests, saved model weights, evaluation plots, dependency instructions, a clean GitHub repository and the university-formatted report. Record actual experiment results rather than inventing accuracy or performance figures.



One technical detail matters: the GRU must actually consume the ten historical traffic observations as a sequence. The other contextual features must be handled separately or integrated appropriately. Feeding every column into an arbitrary neural network and calling it a time-series model would weaken the methodology.
4. Antigravity plugins and MCP servers
I recommend a deliberately small setup. Antigravity supports MCP integrations and project-level rules, so there is no need to install a large collection of unrelated tools. 


Google Antigravity Docs
+1



Context7 MCP — recommended
Use it for current library documentation when working with TensorFlow/Keras, scikit-learn and Streamlit. This reduces errors caused by outdated API examples.

Install


GitHub integration — optional
Useful if you want the agent to interact with your remote repository directly. Git and the terminal are enough for local commits, so do not install this purely for version control.

Optional


Playwright MCP — optional
Useful for browser-level testing of the completed Streamlit interface. We can add it once the application works; it is not needed for downloading data or training the model.

Later


Do not add a vector database, web-scraping infrastructure, paid weather service, cloud database, or multiple overlapping MCP servers. They do not solve a problem this project has.
I would also establish project-level rules so Antigravity works autonomously inside the project folder, downloads data only from specified sources, runs tests and training before declaring success, preserves existing files, and asks before destructive operations or external deployment.
5. Execution schedule
Your assignment document lists 23 October 2026 as the final deadline. As today is 13 October, there are approximately ten days left. 
DOC-20261012-WA0012..pdf

