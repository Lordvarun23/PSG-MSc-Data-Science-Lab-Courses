# KNN-Regressor-Cardekho
Implemented K Nearest Neighbor Regressor. Both using Mean and Median of K NN. Compared the performance of each. Also have implemented cross validation to choose optimal k value.
# K Nearest Neighbors Regression on Cardekho Dataset
<ol>
<li> Instance Based Algorithm (Lazy Learner).</li>
<li> Supervised Learning Algorithms which works for both Classification and Regression. </li>
<li> It works based on Mode of KNN for Classification and Mean for Regression.</li>
<li> It works for both Categorical and Numerical Independet Features. </li>
</ol>

<b> Algorithm: </b>
<ol>
<li> Find Distance of each xtest point to xtrain points. </li>
<li> Choose K nearest neigbours of xtest </li>
<li> Classification -> mode(y1,y2....yk) --> yhat, 
<br> Regression -> mean(y1,y2,..yk) ---> yhat </li>
<li> Check Performance metrics </li>
</ol>

<b>Note: </b>
<ul>
<li> k is the hyperparameter can be choosen using cross validation. </li>
<li> Higher the k more the chance for underfitting </li>
<li> Lower the k value more is the chance for overfitting. </li>
<li> There is impact of outliers depending upon k value </li>
<li> Greaty affected by class imbalance. </li>

</ul>

<b> Basic Preprocessing Steps: </b>
<ul>
<li> As KNN works based on Distace so it is better to standardize or normalize before model building. </li>
<li> Handling Outliers. </li>
<li> Handling Imbalanced Datasets by upsampling and downsampling. </li>

<h4> Hyperparamter Tuning - K </h4>
<img src = "https://user-images.githubusercontent.com/69851775/206198435-ff5968b9-8a70-4075-8c8c-045bf1e14782.png"></img>

<h4> Results of Various Models</h4>
<table>
  <tr>
    <th>Model</th>
    <th>Test RMSLE</th>
  </tr>
  <tr>
    <td>KNN K=19</td>
    <td>12.995</td>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>13.107</td>
  </tr>
  <tr>
    <td>Decision tree</td>
    <td>13.182</td>
  </tr>
  <tr>
    <td>Support Vector Regressor</td>
    <td>13.383</td>
  </tr>
  <tr>
    <td>Random Forest Regressor</td>
    <td>13.057</td>
  </tr>
  <tr>
    <td>XGBoost Regressor</td>
    <td>13.122</td>
  </tr>
</table>

<h4> Results: </h4>
<img src = "https://user-images.githubusercontent.com/69851775/206237358-3172afa9-9120-4fe1-b4bf-ff769155acab.png"></img>
