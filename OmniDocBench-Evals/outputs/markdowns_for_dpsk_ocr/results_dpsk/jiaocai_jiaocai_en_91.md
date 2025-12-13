<|ref|>table<|/ref|><|det|>[[139, 83, 620, 135]]<|/det|>

<table><tr><td>UAT</td><td>C++</td></tr><tr><td>UAT</td><td>Finance</td></tr></table>  

<|ref|>text<|/ref|><|det|>[[147, 140, 850, 175]]<|/det|>
In addition, we define that "an employee fits a job" if and only if the employee has all skills that are needed by the job.  

<|ref|>text<|/ref|><|det|>[[147, 180, 704, 198]]<|/det|>
Please write relational algebra expressions for the following queries.  

<|ref|>text<|/ref|><|det|>[[148, 203, 812, 238]]<|/det|>
1) Find the name of female employees who have at least one skill needed by the "DEV" job.  

<|ref|>text<|/ref|><|det|>[[147, 243, 615, 261]]<|/det|>
2) Find the names of employees who fit the "DEV" job.  

<|ref|>sub_title<|/ref|><|det|>[[147, 268, 553, 288]]<|/det|>
## 2. SQL Query (20 points, 5 points each)  

<|ref|>text<|/ref|><|det|>[[147, 293, 835, 328]]<|/det|>
Consider the relational schemas given in problem 1, please write SQL statements to meet the following requests.  

<|ref|>text<|/ref|><|det|>[[147, 333, 840, 440]]<|/det|>
1) Find the employees who have not any skills.  
2) Find the jobs that need at least "Java" and "C++" skills.  
3) Find the names of employees who have the maximum number of skills among all employees.  
4) Find the employees who fit both the "DEV" and "UAT" jobs.  

<|ref|>sub_title<|/ref|><|det|>[[148, 445, 455, 465]]<|/det|>
## 3. Embedded SQL (10 points)  

<|ref|>text<|/ref|><|det|>[[148, 470, 831, 521]]<|/det|>
Based on the schemas defined in problem 1, the following embedded SQL program accepts the id of an employee as input, and output all skills of the employee. Please fill in the blanks of the program.  

<|ref|>text<|/ref|><|det|>[[148, 528, 210, 544]]<|/det|>
main()  

<|ref|>text<|/ref|><|det|>[[149, 549, 480, 567]]<|/det|>
{ EXEC SQL INCLUDE SQLCA;  

<|ref|>text<|/ref|><|det|>[[207, 573, 568, 590]]<|/det|>
EXEC SQL BEGIN DECLARE SECTION;  

<|ref|>text<|/ref|><|det|>[[208, 597, 430, 614]]<|/det|>
char id[10]; char skill [20];  

<|ref|>text<|/ref|><|det|>[[208, 620, 548, 637]]<|/det|>
EXEC SQL END DECLARE SECTION;  

<|ref|>text<|/ref|><|det|>[[208, 643, 787, 661]]<|/det|>
EXEC SQL CONNECT TO skill_db USER use1 USING password1;  

<|ref|>text<|/ref|><|det|>[[208, 668, 844, 688]]<|/det|>
EXEC SQL DECLARE skill_cursor CURSOR for ①  

<|ref|>text<|/ref|><|det|>[[208, 696, 506, 714]]<|/det|>
printf("please input employee id :");  

<|ref|>text<|/ref|><|det|>[[208, 720, 347, 737]]<|/det|>
scanf("%s", id);  

<|ref|>text<|/ref|><|det|>[[208, 744, 596, 765]]<|/det|>
EXEC SQL ②  

<|ref|>text<|/ref|><|det|>[[211, 774, 275, 791]]<|/det|>
for (;;)  

<|ref|>text<|/ref|><|det|>[[208, 798, 591, 820]]<|/det|>
{ EXEC SQL ③  

<|ref|>text<|/ref|><|det|>[[268, 828, 583, 850]]<|/det|>
if ( ④ ) break;  

<|ref|>text<|/ref|><|det|>[[268, 858, 430, 875]]<|/det|>
printf("%s\n", id);