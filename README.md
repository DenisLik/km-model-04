On July 22, 2022 prediction application based two machine learned models was deployed into heroku server. File name of the applycation is km-model-04, and it was created by FastAPI library for Python.

The application is turned on POST requests and link is https://km-model-04.herokuapp.com/predict.
This application can work with the data within only PMP exam preparetion.
First model presents an ordinary Decision Tree Regressor model and predicts how many exam questions are remain for an user point where there is 99,9% of probability of success PMP exam pass for the user.
Second model presents an ordinary Nearest Neighbors Regressor model and predicts what probability of success PMP exam pass fot the user with overal mean result for the user by cumulative total (current knowledge level).
As input data for the application are json format data and they contains the following parameters:

user_id - identifier of registered user (unregistered user has null id)

temp_id - temporary identifier of an user (registered user has null temporary id)

exam_id - identifier of an exam

try_id - number of the exam attempt

items - array of all passed question ids, answer dates, user answers, rigth answers for the exam id.

input_2656_2.txt
04 Aug 2022, 11:19 AM
 

Example of an input:
{
"user_id": 2656,
"temp_id": 0,
"exam_id": 24,
"try_id": 1,
"items": [{
"question_id": 1343,
"answer_date": "2021-11-05 19:43:58",
"user_answer": "c",
"right_answer": "c"
}, {
"question_id": 1749,
"answer_date": "2021-11-05 20:12:52",
"user_answer": "a",
"right_answer": "a"
}, {
"question_id": 279,
"answer_date": "2021-11-05 20:41:28",
"user_answer": "b",
"right_answer": "d"
},
.................
, {
"question_id": 946,
"answer_date": "2022-02-20 18:25:51",
"user_answer": "c",
"right_answer": "c"
}]
}

As

 from the application are json format data and they contains the following parameters:

amount_of_questions_passed - how many questions are passed at current point in total

amount_of_questions_lost - how many exam questions are remain for an user point where there is 99,9% of probability of success PMP exam pass for the user

overal_mean_result - overal mean result for the user by cumulative total

probability - probability of success PMP exam pass fot the user with overal mean result (current knowledge level)

Example of an output:
{
"amount_of_questions_passed": 55,
"amount_of_questions_lost": 1120,
"overal_mean_result": 67.27,
"probability": 61.63
}

Major application work steps:

The prediction application gets an input from KnowledgeMap web application

The prediction application analizes the input (is it PMP preparation? is user registered? and more)

The prediction application downloads all information about the user from own database

The prediction application calculates main statistic parameters (mean result, overal question, overal attempts and more)

The prediction application calculates two output parameters by machine learning models

The prediction application sends the output data and upgrades own database
