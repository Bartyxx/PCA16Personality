<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<h1 align="center">PCA16Personality</h1><br/>
<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<div>
    <p align="center" dir="auto">
        Calulcate the accuracy of various models on a 16 personality test database.
    </p>
    <div align="center">
        <img src = "https://github.com/Bartyxx/PCA16Personality/blob/main/image/personalitylogo.png"/>
    </div>
</div>
<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<h1>Reference</h1><br/>
<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<i>data/16P.csv</i> --> Dataset used for the following analysis, downloaded at: 'https://www.kaggle.com/datasets/anshulmehtakaggl/60k-responses-of-16-personalities-test-mbt'.<br/>
The dataset is based on the sixteen personality tests. Available here(https://www.16personalities.com/) after answering a series of questions the test divides the users into one of the sixteen personalities.<br/>
<br/>

<table>
<caption>Explenation of the personality type</caption>
<tr><td><b>Personality</b></td><td><b>Type</b>    </td></tr>
<tr><td>ESTJ               </td><td>The Supervisor</td></tr>
<tr><td>ENTJ               </td><td>The Commander </td></tr>
<tr><td>ESFJ               </td><td>The Provider  </td></tr>
<tr><td>ENFJ               </td><td>The Giver     </td></tr>
<tr><td>ISTJ               </td><td>The Inspector </td></tr>
<tr><td>ISFJ               </td><td>The Nurturer  </td></tr>
<tr><td>INTJ               </td><td>The Mastermind</td></tr>
<tr><td>INFJ               </td><td>The Counselor </td></tr>
<tr><td>ESTP               </td><td>The Doer      </td></tr>
<tr><td>ESFP               </td><td>The Performer </td></tr>
<tr><td>ENTP               </td><td>The Visionary </td></tr>
<tr><td>ENFP               </td><td>The Champion  </td></tr>
<tr><td>ISTP               </td><td>The Craftsman </td></tr>
<tr><td>ISFP               </td><td>The Composer  </td></tr>
<tr><td>INTP               </td><td>The Thinker   </td></tr>
<tr><td>INFP               </td><td>The Idealist  </td></tr>
</table>

<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<h1>Data</h1>
<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
The dataset is composed by 62 columns and 59.999 rows.<br/>
The columns are:<br/>
<b>column[Response_Id] </b> --> progressive id.                               <br/>
<b>60 columns of answer</b> --> asnwer of the test.                           <br/>
<b>column[Personality ]</b> --> the resultant personality after take the test.<br/>
<br/>
<table>
<tr><td><b>Response Id</b></td> 
    <td><b>You regularly make new friends.	</b></td>
    <td><b>You are very sentimental.</b></td> 
    <td><b>You enjoy participating in group activities.</b>	</td>
    <td>...</td> 
    <td><b>You struggle with deadlines.</b>	</td>
    <td><b>You feel confident that things will work out for you.</b></td>
    <td><b>Personality</b></td>
                               </tr> 
<tr><td>0</td><td>0</td><td>0</td><td>-1</td><td></td><td>0 </td><td>0</td><td>ENFP</td></tr>
<tr><td>1</td><td>0</td><td>0</td><td>-2</td><td></td><td>-1</td><td>3</td><td>ISFP</td></tr>
<tr><td>2</td><td>0</td><td>0</td><td>2 </td><td></td><td>2 </td><td>1</td><td>INFJ</td></tr>
</table>

Every row represents one user that take the test, and the possible answer to every questions is 7, represented with a number in the following way:<br/>
<table>
<tr><td><b>Value in the dataset</b></td><td><b>Answer in the test</b></td></tr>
<tr><td>3                           </td><td>Strongly Agree          </td></tr>
<tr><td>2                           </td><td>Agree                   </td></tr>
<tr><td>1                           </td><td>Slightly Agree          </td></tr>
<tr><td>0                           </td><td>Neutral                 </td></tr>
<tr><td>-1                          </td><td>Slightly Disagree       </td></tr>
<tr><td>-2                          </td><td>Disagree                </td></tr>
<tr><td>-3                          </td><td>Strongly Agree          </td></tr>
</table>
<br/><br/>
One question in the test:<br/>
<p><img src = "https://github.com/Bartyxx/PCA16Personality/blob/main/image/personality_test_answer.png"/></p>

<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<h1>Data Modeling</h1>
<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
Dataset changes:<br/>
1 - Remove the column [Response Id]<br/>
2 - Code the column [Personality] from <i>str</i> to <i>integer</i> for use the ML models.<br/>

<table>
<caption>Mapping in the dataset</caption>
<tr><td><b>Number in the code</b></td><td><b>Corresponding personality</b></td></tr>
<tr><td>0                        </td><td>ESTJ                            </td></tr>
<tr><td>1                        </td><td>ENTJ                            </td></tr>
<tr><td>2                        </td><td>ESFJ                            </td></tr>
<tr><td>3                        </td><td>ENFJ                            </td></tr>
<tr><td>4                        </td><td>ISTJ                            </td></tr>
<tr><td>5                        </td><td>ISFJ                            </td></tr>
<tr><td>6                        </td><td>INTJ                            </td></tr>
<tr><td>7                        </td><td>INFJ                            </td></tr>
<tr><td>8                        </td><td>ESTP                            </td></tr>
<tr><td>9                        </td><td>ESFP                            </td></tr>
<tr><td>10                       </td><td>ENTP                            </td></tr>
<tr><td>11                       </td><td>ENFP                            </td></tr>
<tr><td>12                       </td><td>ISTP                            </td></tr>
<tr><td>13                       </td><td>ISFP                            </td></tr>
<tr><td>14                       </td><td>INTP                            </td></tr>
<tr><td>15                       </td><td>INFP                            </td></tr>
</table>

<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<h1>File</h1>
<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<i>main.py</i> --> Code used for realize the project.
                  It's divided into two parts:<br/>
                  1 - Try different models in the dataset, the models are:
                  <table>
                  <tr><td><b>Model</b>                 </td><td><b>Accuracy on 10 attempts</b></td></tr>
                  <tr><td>KNN, k = 3                   </td><td> 98.80777777777777%           </td></tr>
                  <tr><td>KNN, k = 5                   </td><td> 98.93777777777778%           </td></tr>
                  <tr><td>KNN, k = 7                   </td><td> 98.94777777777777%           </td></tr>
                  <tr><td>Linear SVM                   </td><td> 94.71000000000001%           </td></tr>
                  <tr><td>Non linear SVM               </td><td> 98.851%                      </td></tr>
                  <tr><td>Neural Network with 10 layers</td><td> 91%                          </td></tr>
                  </table>             
                  2 - MyPCA, in this case, I am removing all the columns that have more than 50.000 zeros and repeating all the models to see the difference.<br/><br/>
                  <table>
                  <caption>Columns with more than 50000 zeros are</caption>
                  <tr><td><b>Name of the colum</b></td> <td><b>Number of Zeros</b></td></tr>
                  <tr><td>You regularly make new friends.                                                                             </td><td>51981</td></tr>
                  <tr><td>You spend a lot of your free time exploring various random topics that pique your interest                  </td><td>52021</td></tr>
                  <tr><td>You are very sentimental.                                                                                   </td><td>51905</td></tr>
                  <tr><td>Even a small mistake can cause you to doubt your overall abilities and knowledge.                           </td><td>51902</td></tr>
                  <tr><td>You avoid leadership roles in group settings.                                                               </td><td>51945</td></tr>
                  <tr><td>You think the world would be a better place if people relied more on rationality and less on their feelings.</td><td>51942</td></tr>
                  <tr><td>You prefer to do your chores before allowing yourself to relax.                                             </td><td>51873</td></tr>
                  <tr><td>You lose patience with people who are not as efficient as you.                                              </td><td>51971</td></tr>
                  <tr><td>You become bored or lose interest when the discussion gets highly theoretical.                              </td><td>51815</td></tr>
                  <tr><td>You usually postpone finalizing decisions for as long as possible.                                          </td><td>52080</td></tr>
                  <tr><td>You rarely contemplate the reasons for human existence or the meaning of life.                              </td><td>51999</td></tr>
                  <tr><td>You take great care not to make people look bad, even when it is completely their fault.                    </td><td>51952</td></tr>
                  <tr><td>When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.       </td><td>51982</td></tr>
                  <tr><td>You would love a job that requires you to work alone most of the time.                                      </td><td>51986</td></tr>
                  <tr><td>You believe that pondering abstract philosophical questions is a waste of time.                             </td><td>51891</td></tr>
                  <tr><td>You know at first glance how someone is feeling.                                                            </td><td>51939</td></tr>
                  <tr><td>You complete things methodically without skipping over any steps.                                           </td><td>52023</td></tr>
                  <tr><td>You are very intrigued by things labelled as controversial.                                                  </td><td>51858</td></tr>
                  </table>
                  <table>
                  <caption>Accuracy after MyPca</caption>
                  <tr><td><b>Model</b>                 </td><td><b>Accuracy on 10 attempts</b></td></tr>
                  <tr><td>KNN, k = 3                   </td><td>98.84%                        </td></tr>
                  <tr><td>KNN, k = 5                   </td><td>98.92%                        </td></tr>
                  <tr><td>KNN, k = 7                   </td><td>98.94%                        </td></tr>
                  <tr><td>Linear SVM                   </td><td>95.07%                        </td></tr>
                  <tr><td>Non linear SVM               </td><td>98.91%                        </td></tr>
                  <tr><td>Neural Network with 10 layers</td><td>90%                           </td></tr>
                  </table>

                  
<i>function.py</i> --> Contain the "<ins>count_unique</ins>" function which is used to count the number of answers of every type. The answers that have more than 50.000 zeros are reported and removed.
                       The "<ins>count_unique</ins>" is imported in "<ins>main.py</ins>".

<i>columns.py</i> --> Contain three variables: <br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <b>columns</b>           : Contain all the columns of the dataset.<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <b>drop_columns</b>      : Columns droppend, the one with more than 50.000 zeros.<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <b>non_drop_columns</b>  : The remaining 42 columns, used for calculate the accuracy with the different model after the removal of the drop_columns.<br/>
                     drop_columns and non_drop_columns are imported and use in: "<ins>main.py</ins>".<br/>

<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->
<h1>Result</h1>
<!-- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ -->

Accuracy of the models, they all above 90%, the higher is the KNN k = 7.<br/>
<p><img src = "https://github.com/Bartyxx/PCA16Personality/blob/main/image/accuracy_10attempts.png"/></p>

Accuracy removing columns, it's nearly higher considering 30 columns but even after considering only 10 columns is acceptable.<br/>
<p><img src = "https://github.com/Bartyxx/PCA16Personality/blob/main/image/%25accuracy.png"/></p>

The information became significant when 42 columns were considered, that is what we were expecting considering that after I removed 18 columns. (The dataset is composed of 60 columns, 42 + 18.)<br/>
<p><img src = "https://github.com/Bartyxx/PCA16Personality/blob/main/image/%25information.png"/></p>

Accuracy after MyPca(removing 18 columns), it's nearly the same as the first models, on average it's nearly -2% so it's acceptable to remove those columns.<br/>
<p><img src = "https://github.com/Bartyxx/PCA16Personality/blob/main/image/accuracy_after_PCA.png"/></p>
