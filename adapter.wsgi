import sys, os
sys.path = ["D:\work\LdosRecSys"] + sys.path
import bottle
os.chdir(os.path.dirname(__file__))
import theRecommenderV2 # This loads your application
application = bottle.default_app()