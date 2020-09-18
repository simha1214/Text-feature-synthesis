# Text-feature-synthesis
## Objective:
  To any classification model all we need is a structured data. But in real world, especially in this new Tech age most of the data available is Text. It may be from emails, internet pages, Twitter feeds, or even script of a popular WEB Series. So now, we cannot feed this text directly into any classification model and we need to extract some structured fields from the raw text such that those features must be sufficient enough to describe your business scenario.
  So Here comes our Feature Synthesis approach which will take your raw text as input along with some list of words which are completely related to Line of Business that you are working on so that the exracted features will get filtered on the basis of how symmetrical that the extracted text phrases are towards your LOB.
  Lets get start understanding the process in detail
  1.	First step is to collect all the text related to each unique id value
      - After collecting all the text, we have to do basic cleansing like removing all special characters, punctuations, duplicate text removal etc..,    
  2.  By using Gensim summarizer, we will be generating a overall summary on entire text for each unique id.  
  3.  By using the phrase extraction techniques rake_nltk and pytext_rank, we will generate text phrases out of all the text of each unique id.  
      - Say suppose from rake_nltk, we got 50 phrases and from pytext_rank we got 40 phrases. On a whole set out of 90 phrases say 30 are duplicate phrases.
      - In this scenario, we are using fuzzywuzzy library to identify and remove duplicate phrases from the whole set by keeping some threshold value of 85(this may vary according to your use case).    
  4.  Now we will generate the symmetricity score in weighted sum fashion as explained below  
      - One is symmetricity score between summary statement and the phrases we finalized at step 3(b).
      - Another symmetricity score is between LOB level list of words and phrases( We fetch the score of maximum symmetrical word and the phrase as final score of the phrase)
      - Now we calculate the weighted sum of these two scores with 75% to summary based score and 25% to list of words score
        - **Score (summary, phrase)*0.75 + Score (max (BoW, phrase)) * 0.25
      - Using this final score, we sort them and keep in use of top 25 phrases from each claim.
  5.  Once the phrase finalization is done, we will now run centroid based clustering technique on top of those phrases. 
  6.  Now we have to penalize the clusters with phrases that will be occurring in most common among all text but that are not too important for our scenario 
      - Here we find out the total cluster occurrence count in the entire corpus by summing up the individual phrase occurrence count inside that cluster.
      - Mean symmetricity score of all phrases inside the cluster w.r.t List of Words defined earlier which are very particular towards our LOB
      - By using below formulae, we finalize the cluster score which penalizes the unwanted clusters.
        - **Occurrence count * 0.25  +  symmetricity_score * 0.75
      - Now we remove clusters whose cluster occurrence count falling above 80th percentile and symmetricity score falling below 90th percentile
  7.  Now the cluster centroids will serve as structured features
  

# Execution
Please do install all the required packages that are mentioned in the requirements.txt

Now in main.py file please change the input data reding command according to your file type
```python
df = pd.read_csv('data/Game_of_Thrones_Script.csv') # this command will change according to the type of input file 
```
After This you have to two variables one will hold your unique key column name and another variable will hold the column name of the text.
```python
unique_key_column_name = 'ID'
text_column_name = 'Sentence'

df.rename(columns={unique_key_column_name:'ID',text_column_name:'Sentence'},inplace=True) 
```
If you have mutilple columns combination as unique key the you can change this below mentioned piece of code which is just below the command to read input data.
```python
df['ID'] = df['Season'] + '_' + df['Episode']# Here we are creating the unique key column out of two individual columns 
```
  Remaining code will run automatically with no human intervention. Only thing needed is if your input size is too big then you might get large number of clusters. For us on 1000 records we got around 2500+ clusters.
  So we went with second level clustering on top of the initially found cluster centroids. In this second level clustering, final cluster count has been reduced to 70+
  
  You can find this optional code in the last of the main.py that looks like below
```python
# Second level of clustering
# ! This is required only if you want to reduce the cluster count which ultimately acts as final features
## OPTIONAL CODE BLOCK ##
#########################

messages = list(clust_df['cluster_name'])
message_embeddings = embed(messages)
# getClusters(message_embeddings)
id2rel = {i:k for i,k in enumerate(messages)}
# rel2embed = {key : getEmbeddings(glove , val) for key,val in id2rel.items()}
rel_clusters = getClusters(message_embeddings)
final_rel_clusters = getRelRep(rel_clusters,message_embeddings)
inv_final_rel = {vv : k for k,v in final_rel_clusters.items() for vv in v }

clust_res_centroids = {i:list(set(final_rel_clusters[i])) for i in final_rel_clusters if len(list(set(final_rel_clusters[i])))>2}

######################################
## OPTIONAL CODE BLOCK ENDS HERE #####
###################################### 
```
  Now your final features will be stored in the fin_model_data_df incase you went for second level clustering or else if you got stopped at first level of clustering itself then you might get the final features in the clust_df
```python
clust_df.to_csv('results/1stlevelcluster.csv',index=False)# This will be your output file if you are going with 1st level clustering 
fin_model_data_df.to_csv('results/2ndlevelcluster.csv',index=False)# This if output file for second level clustering.
```
Hope you guys found this as useful...:smile:


