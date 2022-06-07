DESCRIPTION
Javascript (project.html)
D3 was used to design interactive choropleth maps of potental offshore wind sites in the United States (US) and total energy consumption
of each state by year. The user can change the year with the dropdown option to see how energy consumption changes through the years along with
a tool tip used to hover over each state to get the exact value of the energy consumption. The user can also choose a clustering method to see all
and change the parameters to see how the offshore wind sites would be clustered by different unsupervised learning methods. Finally, the user can
select the state, % of energy consumption needed to be met, and year and see the optimal location of potential offshore wind sites.

Tkinter GUI (project_cluster_GUI.py)
NOTE: We have had issues with the GUI working on Apple devices (Macbook) the GUI works and is usable on the rest of our teams' windows machines.
Tkinter is a Python binding to the Tk GUI toolkit. It is Python's standard GUI. This tool allows users to interact with our clustering algorithms and 
test out different parameters and save their results. The tool allows the user to have some decision making in their choice of clustering algorithm and 
the way they want the wind sites data to be clustered. Each model will provide visuals of the user-defined clustering model and metrics that evaluate the model.   

Juypter Notebooks (get_powerplant_netgeneration_bystate_byyear.ipynb & clustering_interactive_plots.ipynb) 
The Jupyter Notebook is an open source web application that you can use to create and share documents that contain live code, equations, visualizations, and text.
We have two jupyter notebooks that are used to gather data for our project and visualize results created by the user. We gather data using API's and parsing through
a JSON response. We use packages such as plotly (scatter_geo) that allows the latitude/longitude and cluster data to be mapped to a dynamic map of the US that the
user can use to zoom in and examine individual regions/data points. 


INSTALLATION
Javascript Directions
Make sure you have the following packages on your computer
1. D3.js.library Version 5 (included in the lib folder)
2. Chrome v92.0 (or higher)
3. Python http server

Python and Juypter Directions
Make sure you have something like Anaconda Navigator downloaded on your computer that can intepret .py and ipynb files.
1. Juypter Notebooks Version 6.3.0 or higher (.iypnb) 
2. Spyder Version 4.2.5 (.py)

EXECUTION
Javascript Directions
1. Open a terminal window and navigate to the directory
2. Execute command to start the server
	Python 2 — python -m SimpleHTTPServer 8000
	Python 3 — python -m http.server 8000
3. Open a web browser at http://localhost:8000/.
4. Click project.html to start visualization

GUI Directions
1. Open Command Prompt and Set Working Directory
2. In Command Prompt type "python project_cluster_GUI.py
	2a. A GUI should appear called "Wind Power Clustering Tool"
3. Click on Green "Import Excel File" and navigate to data >> offshore-wind-open-access-siting-regime-atb-mid-turbine-fy21.csv and select file
4. Try your own test cases for each clustering algorithm by inputing parameters into white entry boxes and clicking corresponding red buttons.
	4a. New windows will appear providing metrics and visualizations of your user defined clusterings. 
	4b. NOTE: DBscan requires both parameters Min Samples and Neighbors Max Distance (Epsilon) to execute. 
5. When done click blue "Save Data" file and save the test cases to be used for US map visualization in clustering_interactive_plots.ipynb (OPTIONAL ***)

Juypter Directions
A. get_powerplant_netgeneration_bystate_byyear.ipynb
	1. Launch Juypter
	2. Navigate to directory and open file. 
	3. Run ONLY the first 4 cells 
		3a. The data the file saves at the end has been read in so that the user can visualize the power plants map that was pulled from the EIA's API.
		3b. Play around with hover tool to get information on individual sites (Feel free to zoom in and out to examine the data closer and see where it is located) 
	4. Run the remaining cells (OPTIONAL DO NOT RECOMMEND*****)
		4a. If getting errors with API calls go to https://www.eia.gov/opendata/register.php and register for new API key (they expire every ~90 days)
		4b. We recommend not testing the whole notebook if you don't want to, the api calls a long time and this file is solely built for the purposes of gathering data used in the rest of the project. 
B. clustering_interactive_plots.ipynb
	1. Launch Juypter
	2. Navigate to directory and open file.
	3b. If using your own test cases: make sure to have correct file directories as your data saved from the GUI will not be saved to data folder
	3a. If using the test cases provided: run all cells and view your clustered maps. Map is dynamic can zoom in and out and hover over points for information

DEMO VIDEO:
https://youtu.be/oE1ecC4PwCs