####### ExcelWriter Example 1 ##############
import pandas as pd
import string

with pd.ExcelWriter('test.xlsx') as writer:
	df1.to_excel(writer, sheet_name='sheet1', index=False)
	worksheet1 = writer.sheets['sheet1']
	worksheet1.set_column('A:A',10)
	worksheet1.set_column('B:K',20)
	worksheet1.set_column('L:L',50)
	worksheet1.set_zoom(70)

if len(df2)>0:
	df2.to_excel(writer, sheet_name='sheet2', index=False)
	worksheet2 = writer.sheets['sheet2']
	worksheet2.set_column('A:H',15)
	worksheet2.set_zoom(70)
	worksheet2.hide()
else:
	pass   

######### ExcelWriter Example 2 ###########
import pandas as pd
import string 

excel_cols =  list(string.ascii_uppercase) # create a list of Alphabet letters for cols 

with pd.ExcelWriter(file+ '.xlsx') as writer:
  df_df.to_excel(writer, sheet_name='Sheet1', index=False)
  worksheet1 = writer.sheets['Sheet1']
  for i in excel_cols[0:4]:
      worksheet1.column_dimensions[i].width = 20
  for i in excel_cols[4]:
      worksheet1.column_dimensions[i].width = 57
  for i in excel_cols[5:6]:
      worksheet1.column_dimensions[i].width = 20
  for i in excel_cols[7]:
      worksheet1.column_dimensions[i].width = 50
  for i in excel_cols[8:14]:
      worksheet1.column_dimensions[i].width = 25
  for i in excel_cols[14]:
      worksheet1.column_dimensions[i].width = 100
  worksheet1.sheet_view.zoomScale = 75
  worksheet1.sheet_state = 'visible'  # change to worksheet2.sheet_state = 'hidden' to hide   
  
# Move all Excel files to another directory  
os.system('mv *.xlsx $PWD/"folder/"/')	