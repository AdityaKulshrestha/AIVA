import win32com.client

# Create a connection to the Windows Search COM object
search = win32com.client.Dispatch("Search.QueryHelper")

# Define your search query
query = "SELECT System.ItemPathDisplay FROM SystemIndex WHERE System.FileName='report.pdf'"

# Execute the search
results = search.Execute(query)

# Print the results
for result in results:
    print(result)