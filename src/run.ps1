# rm *.class minet/*.class minet/*/*.class

# complie all the .java files
echo "Compiling..."
javac -cp lib/*:minet:. minet/*.java minet/*/*.java *.java