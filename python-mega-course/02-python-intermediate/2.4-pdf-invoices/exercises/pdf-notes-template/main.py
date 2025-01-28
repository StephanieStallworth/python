from fpdf import FPDF # class that creates PDF instances
import pandas as pd

# PDF instance with no pages
pdf = FPDF(orientation="P", unit="mm", format="A4")

# Configure main document
# So pages are not broken automatically
pdf.set_auto_page_break(auto=False, margin=0)

# Add one page per topic
df = pd.read_csv("topics.csv")

for index,row in df.iterrows(): # gives us access to every row in the dataframe and its index
    pdf.add_page()

    ########## Set the header for the master page ##########
    pdf.set_font(family="Times", style="B", size=24)
    pdf.set_text_color(100,100,100) # RGB combination for grey, can provide any number 0-254 for each of the values to get different colors
    pdf.cell(w=0, h=12, txt=row["Topic"], align="L",  ln=1)

    # Create a line and provide coordinates for it
    # x1 and y1 are coordinates for start point, x2 and y2 are coordinates of end point
    pdf.line(x1 = 10, # distance from the left border to this point in millimeters (units we defined in the FPDF line)
             y1 = 21, # distance from the top to this point in millimeters
             x2 = 200,
             y2 = 21 # has to be same as y1 to have a straight line
             )

    ########## Set the footer for the master page ##########
    # Empty breaklines, specify how many millimeters down the page footer should be
    # Height of A4 format is 298 millimeters
    # If value is too big, won't fit on page
    pdf.ln(265)

    pdf.set_font(family="Times", style="I", size=8)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(w=0, h=10, txt=row["Topic"], align="R")

    ########## Creating multiple pages per topic ##########
    # Subtracting one from value because the parent/master page was already created above for each topic containing the header
    for i in range(row["Pages"]-1):
        pdf.add_page()

        # Set footer for sub-pages
        pdf.ln(277) # 12 (height of header cell) + 265
        pdf.set_font(family="Times", style="I", size=8)
        pdf.set_text_color(180, 180, 180)
        pdf.cell(w=0, h=10, txt=row["Topic"], align="R")

########## Final Output ##########
# Convert Python object to an actual PDF file on disk
pdf.output("output.pdf")