# Fraud Detection ML System

A production-grade fraud detection system designed to match senior data scientist role requirements. This project implements end-to-end fraud detection with XGBoost and Isolation Forest models, comprehensive feature engineering, real-time API deployment, and data drift monitoring.

## 🎯 Project Highlights

✅ **Data Integration**: Integrated synthetic credit card transactions with user demographics  
✅ **Feature Engineering**: Created 13+ behavioral, velocity, and geospatial features  
✅ **Model Development**: XGBoost (cost-weighted), XGBoost (SMOTE), and Isolation Forest  
✅ **Business Metrics**: Precision-Recall AUC, financial impact analysis  
✅ **Production API**: FastAPI microservice for real-time predictions  
✅ **Drift Monitoring**: KS-test and statistical distribution monitoring  
✅ **Unit Tests**: Comprehensive data validation tests  
✅ **Docker**: Containerized deployment-ready application  

## 📁 Project Structure

```
fraud-detection-ml/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data sourcing and integration
│   ├── feature_engineering.py  # Behavioral, velocity, geospatial features
│   ├── model_trainer.py        # XGBoost and Isolation Forest training
│   ├── model_evaluator.py      # Business-focused evaluation metrics
│   ├── drift_monitor.py        # Production data monitoring
│   └── utils.py                # Utility functions
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   └── schemas.py              # Pydantic models
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   └── test_model_evaluation.py
├── models/                     # Saved model artifacts
├── notebooks/
│   └── fraud_analysis.ipynb
├── Dockerfile
├── requirements.txt
├── config.yaml
├── main.py                     # Execute full pipeline
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/jay154/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2. Set Up Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Full Pipeline
```bash
python main.py
```

This will:
- Load and integrate synthetic datasets
- - Engineer 13+ features (velocity, ratio, geospatial, behavioral)
  - - Train 3 models with class imbalance handling (SMOTE, cost-weighting)
    - - Evaluate models using business metrics (Precision-Recall AUC)
      - - Save trained models to `models/`
       
        - ### 4. Run Unit Tests
        - ```bash
          pytest tests/ -v
          ```

          ### 5. Start FastAPI Server
          ```bash
          python -m uvicorn app.main:app --reload --port 8000
          ```

          Visit `http://localhost:8000/docs` for interactive API documentation.

          ## 📊 Key Features

          ### Data Integration & Cleaning
          - Synthetic credit card transactions (50,000 records)
          - - User demographics integration
            - - Data validation (no negative amounts, valid timestamps, etc.)
              - - 2% fraud rate (~1,000 frauds in 50k transactions)
               
                - ### Feature Engineering
                - **Velocity Features** (capture emerging fraud patterns):
                - - `txn_count_1h` - Transactions in last hour
                  - - `txn_count_24h` - Transactions in last 24 hours
                   
                    - **Ratio Features** (compare to user's behavior):
                    - - `amount_to_avg_ratio` - Current amount vs. user's average daily spend
                      - - `amount_to_median_ratio` - Current amount vs. median transaction
                        - - `amount_zscore` - Standard deviations from user's mean spending
                         
                          - **Geospatial Features** (detect rapid movements):
                          - - `distance_from_last_txn` - Distance from last transaction location
                            - - `category_mismatch` - Transaction differs from user's preferred merchant
                              - - `unusual_hour` - Transaction at unusual time for user
                               
                                - **Account Features**:
                                - - `account_age_years` - Account maturity
                                  - - `is_business` - Business vs. individual account
                                    - - `is_domestic` - Domestic vs. international
                                     
                                      - ### Model Development
                                     
                                      - #### XGBoost with Cost-Weighting
                                      - - Handles extreme class imbalance (1:1000)
                                        - - Higher cost for false negatives (missed fraud)
                                          - - 100 estimators, max_depth=7
                                           
                                            - #### XGBoost with SMOTE
                                            - - Synthetic oversampling of minority class
                                              - - Improved recall for fraud detection
                                                - - k_neighbors=3 for feature-space interpolation
                                                 
                                                  - #### Isolation Forest
                                                  - - Unsupervised anomaly detection
                                                    - - Contamination = 0.02 (expected fraud rate)
                                                      - - Captures unexpected patterns beyond supervised training
                                                       
                                                        - ### Evaluation Metrics
                                                        - - **Precision-Recall AUC**: Primary metric for imbalanced data
                                                          - - **ROC AUC**: Threshold-agnostic performance
                                                            - - **F1-Score**: Harmonic mean at different thresholds
                                                              - - **Business Impact**: Dollar savings vs. customer friction costs
                                                               
                                                                - Example confusion matrix at 0.5 threshold:
                                                                - - TP: True frauds caught (financial saved)
                                                                  - - FP: Legitimate txns blocked (customer loss)
                                                                    - - TN: Legitimate txns allowed
                                                                      - - FN: Frauds missed (financial loss)
                                                                       
                                                                        - ### Production Deployment
                                                                       
                                                                        - **FastAPI Endpoints**:
                                                                        - ```bash
                                                                          GET /health                    # Health check
                                                                          POST /predict                  # Single prediction
                                                                          POST /predict_batch            # Batch predictions
                                                                          ```

                                                                          **Example Request**:
                                                                          ```bash
                                                                          curl -X POST "http://localhost:8000/predict" \
                                                                            -H "Content-Type: application/json" \
                                                                            -d '{
                                                                              "amount": 500.00,
                                                                              "txn_count_1h": 3,
                                                                              "txn_count_24h": 12,
                                                                              "amount_to_avg_ratio": 3.5,
                                                                              "amount_to_median_ratio": 2.1,
                                                                              "amount_zscore": 2.3,
                                                                              "distance_from_last_txn": 1500,
                                                                              "category_mismatch": 1,
                                                                              "unusual_hour": 0,
                                                                              "account_age_years": 2.5,
                                                                              "is_business": 0,
                                                                              "is_domestic": 1,
                                                                              "avg_daily_spend": 150.00
                                                                            }'
                                                                          ```

                                                                          **Response**:
                                                                          ```json
                                                                          {
                                                                            "fraud_probability": 0.87,
                                                                            "is_fraud": true,
                                                                            "risk_level": "HIGH",
                                                                            "timestamp": "2024-01-06T04:59:22Z"
                                                                          }
                                                                          ```

                                                                          ### Drift Monitoring

                                                                          Detects distribution shifts in production data using:
                                                                          - **Kolmogorov-Smirnov Test**: Sensitive to distribution changes
                                                                          - - **Z-Test**: Detects mean/variance shifts
                                                                            - - **Fraud Rate Monitoring**: Tracks relative changes in fraud prevalence
                                                                             
                                                                              - Triggers alerts when:
                                                                              - - p-value < 0.05 (significant distribution shift)
                                                                                - - Fraud rate changes >20% relative
                                                                                 
                                                                                  - ## 🐳 Docker Deployment
                                                                                 
                                                                                  - ### Build Docker Image
                                                                                  - ```bash
                                                                                    docker build -t fraud-detection-api .
                                                                                    ```

                                                                                    ### Run Container
                                                                                    ```bash
                                                                                    docker run -p 8000:8000 fraud-detection-api
                                                                                    ```

                                                                                    Health check: `curl http://localhost:8000/health`

                                                                                    ## 🧪 Testing

                                                                                    Run comprehensive unit tests:
                                                                                    ```bash
                                                                                    pytest tests/test_data_validation.py -v
                                                                                    pytest tests/test_feature_engineering.py -v
                                                                                    pytest tests/test_model_evaluation.py -v
                                                                                    ```

                                                                                    Tests validate:
                                                                                    - No negative transaction amounts
                                                                                    - - Reasonable fraud rate (0.5-5%)
                                                                                      - - Valid timestamps and geolocation
                                                                                        - - Account age non-negativity
                                                                                         
                                                                                          - ## 📈 Resume Highlights
                                                                                         
                                                                                          - This project demonstrates:
                                                                                         
                                                                                          - 1. **"Integrated and cleaned disparate datasets"**
                                                                                            2.    - Combined credit card transactions with user demographics
                                                                                                  -    - Validated data quality and handled missing values
                                                                                                   
                                                                                                       - 2. **"Engineered velocity and behavioral features"**
                                                                                                         3.    - Created time-window features capturing fraud patterns
                                                                                                               -    - Implemented geospatial distance calculations
                                                                                                                    -    - Built ratio features comparing to user baselines
                                                                                                                     
                                                                                                                         - 3. **"Iterated on multiple architectures"**
                                                                                                                           4.    - XGBoost with cost-sensitive learning
                                                                                                                                 -    - XGBoost with SMOTE oversampling
                                                                                                                                      -    - Isolation Forest for unsupervised detection
                                                                                                                                           -    - Handled extreme class imbalance (1:1000)
                                                                                                                                            
                                                                                                                                                - 4. **"Optimized for Precision-Recall AUC"**
                                                                                                                                                  5.    - Business-focused metrics (financial impact)
                                                                                                                                                        -    - Threshold optimization for user experience
                                                                                                                                                             -    - Detailed confusion matrix analysis
                                                                                                                                                              
                                                                                                                                                                  - 5. **"Deployed via Dockerized FastAPI microservice"**
                                                                                                                                                                    6.    - RESTful API with Pydantic validation
                                                                                                                                                                          -    - Batch prediction support
                                                                                                                                                                               -    - Health check endpoints
                                                                                                                                                                                    -    - Production-ready error handling
                                                                                                                                                                                     
                                                                                                                                                                                         - 6. **"Implemented data drift monitoring"**
                                                                                                                                                                                           7.    - KS-test for distribution shifts
                                                                                                                                                                                                 -    - Statistical testing for mean/variance changes
                                                                                                                                                                                                      -    - Fraud rate pattern detection
                                                                                                                                                                                                           -    - Automated alerts for model retraining
                                                                                                                                                                                                            
                                                                                                                                                                                                                - 7. **"Comprehensive unit testing"**
                                                                                                                                                                                                                  8.    - Data validation tests
                                                                                                                                                                                                                        -    - Feature engineering tests
                                                                                                                                                                                                                             -    - Model evaluation tests
                                                                                                                                                                                                                                  -    - pytest integration
                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                       - ## 🔧 Configuration
                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                       - Edit `config.yaml` to customize:
                                                                                                                                                                                                                                       - - Model hyperparameters
                                                                                                                                                                                                                                         - - Feature selection
                                                                                                                                                                                                                                           - - Drift detection thresholds
                                                                                                                                                                                                                                             - - API settings
                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                               - ## 📚 Learning Resources
                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                               - - XGBoost documentation: https://xgboost.readthedocs.io/
                                                                                                                                                                                                                                                 - - SMOTE for imbalanced learning: https://imbalanced-learn.org/
                                                                                                                                                                                                                                                   - - FastAPI: https://fastapi.tiangolo.com/
                                                                                                                                                                                                                                                     - - Drift detection: https://evidentlyai.com/
                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                       - ## 💡 Key Takeaways
                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                       - This project showcases:
                                                                                                                                                                                                                                                       - - **End-to-end ownership**: Problem framing → deployment → monitoring
                                                                                                                                                                                                                                                         - - **Practical insight**: Business-focused metrics over academic novelty
                                                                                                                                                                                                                                                           - - **Production quality**: Testing, containerization, monitoring
                                                                                                                                                                                                                                                             - - **Real-time decisioning**: Sub-millisecond API predictions
                                                                                                                                                                                                                                                               - - **Domain expertise**: Understanding fraud patterns and risk operations
                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                 - ## 🤝 Contributing
                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                 - Feel free to extend this project by:
                                                                                                                                                                                                                                                                 - - Adding new feature engineering techniques
                                                                                                                                                                                                                                                                   - - Implementing additional anomaly detection methods
                                                                                                                                                                                                                                                                     - - Enhancing drift monitoring with more statistical tests
                                                                                                                                                                                                                                                                       - - Building a dashboard for model monitoring
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                         - ## 📄 License
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                         - This project is open source and available under the MIT License.
                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                         - ---
                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                         **Ready to use?** Start with `Quick Start` → `Run the Full Pipeline` → `Start FastAPI Server`
