1) produce the projectname.txt - this contains the structured info on your entire script, see examples
for detail
2) stage the input script and validate it, the project file folder should now exist.
3) use the orchestrator to - python orchestrator projectname --autoprocess-all  
    to run the project through all the stages

1) alternatively, make the input.txt script and put that inside the staging folder 
and make sure its the only there
2) do , python orchestrator, this automatically validates and runs it through the pipeline until it creates the video

NOTE: First method allows you to change the config.json for settings like voice type/effects
Second method just uses the default provided in the validator python script