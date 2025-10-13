docker run 	--rm -it -p 8888:8888		^
		-v %CD%\work:/home/kincancer/work 	^
		-v %CD%/..:/kincancer		^
		-v %CD%/../../smolfem:/smolfem	^
		human_kinetic_cancer_docker %*
