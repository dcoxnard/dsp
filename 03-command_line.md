# Learn command line

Please follow and complete the free online [Command Line Crash Course
tutorial](https://web.archive.org/web/20160708171659/http://cli.learncodethehardway.org/book/) or [Codecademy's Learn the Command Line](https://www.codecademy.com/learn/learn-the-command-line). These are helpful tutorials. Each "chapter" focuses on a command. Type the commands you see in the _Do This_ section, and read the _You Learned This_ section. Move on to the next chapter. You should be able to go through these in a couple of hours.

---

### Q1.  Cheat Sheet of Commands  

Here's a list of items with which you should be familiar:  
* show current working directory path
* creating a directory
* deleting a directory
* creating a file using `touch` command
* deleting a file
* renaming a file
* listing hidden files
* copying a file from one directory to another

Make a cheat sheet for yourself: a list of at least **ten** commands and what they do.  (Use the 8 items above and add a couple of your own.)  

`ls` list what's in the current directory
`cd` change working directory
`pwd` print working directory
`rmdir` delete a directory
`touch path/file.ext` create a file with the stated path
`rm` delete file
`mv` rename file
`cp` copy file
`ls -a` list hidden files
`xargs` execute arguments

---

### Q2.  List Files in Unix   

What do the following commands do:  
`ls`  list what's in the current directory
`ls -a`  list hidden files
`ls -l`  list the contents of the directory in long form, with permissions
`ls -lh`  list directory with human readable file size
`ls -lah` combo of the above: list hidden files with human readable form
`ls -t`  list directory sorted by time and date
`ls -Glp`  colorized output, long form, and write a slash '/' if a directory

> > REPLACE THIS TEXT WITH YOUR RESPONSE

---

### Q3.  More List Files in Unix  

Explore these other [ls options](http://www.techonthenet.com/unix/basic/ls.php) and pick 5 of your favorites:

`ls -C` multi-column output
`ls -c` sort by time last modified
`ls -i` print file's serial number
`ls -m` separate files by comma (stream files)
`ls -R` recursively list subdirectories found

---

### Q4.  Xargs   

What does `xargs` do? Give an example of how to use it.

> > `xargs` is a way to pass arguments to a command.  This is useful if you have many inputs for the command, because instead of calling the command once for each input, you can use `xargs` to tell the command to run against each input that you specify.
 

