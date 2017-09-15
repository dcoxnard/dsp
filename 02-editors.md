# Choose and learn your editor(s)


Computing's interface is text. To work effectively, you need to be fluent with this interface.


## Typing

It may sound silly, but make sure you know how to type.  Take this [typing test](http://www.typingtest.com/). You should be comfortable typing with perfect accuracy at 60 words per minute, at least. If you currently can't, practice until you can.

A lot of your work will be done in a text editor. You have to know how to use your editor. Any editor will work, but knowing a powerful editor well will make you faster, more comfortable, and more effective.

---

## Terminal Editor

You will need to use a terminal text editor at times.  You will **always need to use a terminal editor on the cloud** (e.g. Amazon Web Services).  This means an editor that will run entirely inside a terminal window, without spawning a new window, entirely without mouse input.

There are 3 widely used terminal editors.  Here's a [discussion comparing **nano**, **vim**, and **emacs**](http://askubuntu.com/questions/804/comparizon-between-text-editors-in-ubuntu-vim-vs-emacs-vs-nano).  
 * [nano](http://staffwww.fullcoll.edu/sedwards/Nano/IntroToNano.html) - this is the simplest to use 
 * [vim](http://www.howtogeek.com/102468/a-beginners-guide-to-editing-text-files-with-vi/) - is the default on many systems and you might find yourself in it even if you did not intend to
 * [emacs](http://ocean.stanford.edu/research/quick_emacs.html) - oldest editor, has steep learning curve, powerful, has lots of extensible options
 
Note:  Both Emacs and vim have built-in interactive tutorials that you can try.

You should know how to do the following tasks, *at the minimum*, on **nano** and **vim**:  
1.  open a file  
2.  edit a file  
3.  save a file  
4.  exit a file   

Here's a basic list of commands:  

|   |terminal editor | open a file    |  edit a file |  save a file       |  exit a file   |  
|---|----------------|----------------|--------------|--------------------|----------------|
| 1 | nano           | nano filen.py  | [just type]  | ctrl + o, enter    | ctrl + x       |
| 2 | vim, vi        | vim filen.py   | [just type]  | esc :w, enter      | esc :q, enter  |
| 3 | emacs          | emacs filen.py | [just type]  | ctrl-x, xtrl-s     | ctrl-x, ctrl-c |
 

---

## Graphical Editor

You will probably spend most of your time with access to a graphical interface, where you have more choices in editors and integrated development environments.

Popular editors and IDEs include:

 * Emacs
 * vim
 * Sublime
 * Atom
 * Spyder
 * PyCharm
 * [Rodeo](http://blog.yhat.com/posts/introducing-rodeo.html)

If you choose Emacs or vim, you will have essentially the same editor experience across graphical and non-graphical environments, which is nice. It's also nice to be able to work without ever having to use a mouse. Emacs and vim have somewhat steep learning curves, but they give you the ability to customize your environment quite a lot to make it exactly what you want.

Sublime is probably the most popular editor for new coders. You can set it up to integrate with Python fairly well. Atom is pretty similar to Sublime but has an interesting open architecture and is developed by folks at GitHub.

Spyder and PyCharm are IDEs for Python. They try to give you a fully configured setup out of the box.

We will also use Jupyter (IPython) notebooks, but this does not remove the need for proficiency in an editor or IDE.

---

### Q1:  Create Files with Terminal Editors

1.  Create a file called `nutmeg.py`, using **nano**, add a couple of lines of text, save and exit.  
Some sample lines:  
```
print("This file was created using the nano editor")
print("This file was created by Reshama")
```
2.  Create a file called `vinegar.py` using **vim**, add a couple of lines of code, save and exit.
3.  Create a file called `egg.py` using **emacs**, add a couple of lines of code, save and exit.

Upload these 3 files to the [`editors`](editors/) folder.  

### Q2. Terminal Editor

What terminal editor will be your preferred choice of use? How did you make your decision?

>> I decided that I will use vim as my terminal editor.  It comes with a steep learning curve but it is a powerful editor once you know how to use it and are comfortable with it.

--

### Q3. Graphical Editor

What graphical editor will you use? How did you make your decision? What are some interesting features of your editor? What are some useful keyboard shortcuts for your editor? How do you customize your editor?

>> I decided that I will use Spyder as my graphical editor.  I especially like the object lookup and syntax coloring.  I decided on Spyder since it is the out-of-the-box IDE with Anaconda.

 
