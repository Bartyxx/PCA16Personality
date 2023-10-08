<h1>File</h1><br/>

<i>main.py</i> --> Code used for realize the project.
                  It's divided in two part:
                  1 - Try different models in the dataset, the models are:<br/>   
                  <table>
                  <tr><td><b>Model</b>                 </td><td><b>Accuracy on 10 attempts</b></td></tr>
                  <tr><td>KNN, k = 3                   </td><td> 98.80777777777777%           </td></tr>
                  <tr><td>KNN, k = 5                   </td><td> 98.93777777777778%           </td></tr>
                  <tr><td>KNN, k = 7                   </td><td> 98.94777777777777%           </td></tr>
                  <tr><td>Linear SVM                   </td><td> 94.71000000000001%           </td></tr>
                  <tr><td>Non linear SVM               </td><td> 98.851%                      </td></tr>
                  <tr><td>Neural Network with 10 layers</td><td> 91%                          </td></tr>
                  </table>             
                  2 - MyPCA, in this case I am removing all the columns that have more than 50.000 zeros and repeat all the models for see the difference.
                  <table>
                  <tr><td><b>Model</b>                 </td><td><b>Accuracy on 10 attempts</b></td></tr>
                  <tr><td>KNN, k + 3                   </td><td>98.84%                        </td></tr>
                  <tr><td>KNN, k = 5                   </td><td>98.92%                        </td></tr>
                  <tr><td>KNN, k = 7                   </td><td>98.94%                        </td></tr>
                  <tr><td>Linear SVM                   </td><td>95.07%                        </td></tr>
                  <tr><td>Non linear SVM               </td><td>98.91%                        </td></tr>
                  <tr><td>Neural Network with 10 layers</td><td>90%                           </td></tr>
                  </table>








                  
<i>function.py</i> --> Contain the "<ins>count_unique</ins>" function wich is used for count the number of answer of every type. The answer that have more than 50.000 zeros are repoorted and after 
                        removed. The "<ins>count_unique</ins>" is imported in "<ins>main.py</ins>".

<i>columns.py</i> --> Contain three variables: <br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <b>columns</b>           : Contain all the columns of the dataset.<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <b>drop_columns</b>      : Columns droppend, the one with more than 50.000 zeros.<br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <b>non_drop_columns</b>  : The remaining 42 columns, used for calculare the accuracy with the different model after the removal of the drop_columns.<br/>
                     drop_columns and non_drop_columns are importend and use in: "<ins>main.py</ins>".<br/>




<br/><br/><br/>
<table>
<caption>Columns with more than 50000 zeros</caption>
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
<tr><td>You are very intrigued by things labeled as controversial.                                                  </td><td>51858</td></tr>
</table>

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
