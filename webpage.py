from wsgiref.simple_server import make_server

# this will return a text response
def hello_world_app(environ, start_response):
    status = '200 OK'  # HTTP Status
    headers = [('Content-type', 'text/plain')]  # HTTP Headers
    start_response(status, headers)
    return ["Hello World"]

# first argument passed to the function
# is a dictionary containing CGI-style environment variables 
# second argument is a function to call
# make a server and turn it on port 8000
httpd = make_server('', 8001, hello_world_app)
httpd.serve_forever()