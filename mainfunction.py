{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5f6069e7-695b-44bc-928e-9b0e0dc05598",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "Spyder Editor\n",
    "Author : Mwangi\n",
    "Purpose : a simple main example \n",
    "\"\"\"\n",
    "print ('is this a main function example ?') #this code will be executed first \n",
    "\n",
    "def main():\n",
    "    print('yes it is main method function') # then this will be excuted second \n",
    "    \n",
    "# for the code to excute the main method then a condition is given i.e.,\n",
    "    \n",
    "if __name__ == '__main__':\n",
    "    main()\n",
    "    \n",
    "# the mainfunction can be imported in another program file i.e\n",
    "# import mainfunction\n",
    "# note this will call print ('is this a main function example ?')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b4d0b3a3-9466-4f7b-92f2-70d30d900bf2",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'null' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 9\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;124;03mSpyder Editor\u001b[39;00m\n\u001b[1;32m      4\u001b[0m \u001b[38;5;124;03mAuthor : Mwangi\u001b[39;00m\n\u001b[1;32m      5\u001b[0m \u001b[38;5;124;03mPurpose : this program imports mainfunction \u001b[39;00m\n\u001b[1;32m      6\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m      7\u001b[0m \u001b[38;5;66;03m# note this will call print ('is this a main function example ?') first and \u001b[39;00m\n\u001b[1;32m      8\u001b[0m \u001b[38;5;66;03m# print (\"call mainfunction \") second\u001b[39;00m\n\u001b[0;32m----> 9\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mmainfunction\u001b[39;00m\n\u001b[1;32m     10\u001b[0m \u001b[38;5;28mprint\u001b[39m (\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcall mainfunction \u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[0;32m~/Documents/GitHub/Code_3_Camp/mainfunction.py:5\u001b[0m\n\u001b[1;32m      1\u001b[0m {\n\u001b[1;32m      2\u001b[0m  \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcells\u001b[39m\u001b[38;5;124m\"\u001b[39m: [\n\u001b[1;32m      3\u001b[0m   {\n\u001b[1;32m      4\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcell_type\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcode\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m----> 5\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mexecution_count\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[43mnull\u001b[49m,\n\u001b[1;32m      6\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mid\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m5f6069e7-695b-44bc-928e-9b0e0dc05598\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m      7\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmetadata\u001b[39m\u001b[38;5;124m\"\u001b[39m: {},\n\u001b[1;32m      8\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124moutputs\u001b[39m\u001b[38;5;124m\"\u001b[39m: [],\n\u001b[1;32m      9\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124msource\u001b[39m\u001b[38;5;124m\"\u001b[39m: [\n\u001b[1;32m     10\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m# -*- coding: utf-8 -*-\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     11\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     12\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSpyder Editor\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     13\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mAuthor : Mwangi\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     14\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mPurpose : a simple main example \u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     15\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     16\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mprint (\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mis this a main function example ?\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m) #this code will be executed first \u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     17\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     18\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdef main():\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     19\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    print(\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124myes it is main method function\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m) # then this will be excuted second \u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     20\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    \u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     21\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m# for the code to excute the main method then a condition is given i.e.,\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     22\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    \u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     23\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mif __name__ == \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m__main__\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     24\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    main()\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     25\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    \u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     26\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m# the mainfunction can be imported in another program file i.e\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     27\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m# import mainfunction\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     28\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m# note this will call print (\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mis this a main function example ?\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m)\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m     29\u001b[0m    ]\n\u001b[1;32m     30\u001b[0m   }\n\u001b[1;32m     31\u001b[0m  ],\n\u001b[1;32m     32\u001b[0m  \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmetadata\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m     33\u001b[0m   \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mkernelspec\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m     34\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdisplay_name\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mPython 3 (ipykernel)\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     35\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mlanguage\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpython\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     36\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mname\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpython3\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m     37\u001b[0m   },\n\u001b[1;32m     38\u001b[0m   \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mlanguage_info\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m     39\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcodemirror_mode\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m     40\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mname\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mipython\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     41\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mversion\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m3\u001b[39m\n\u001b[1;32m     42\u001b[0m    },\n\u001b[1;32m     43\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mfile_extension\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m.py\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     44\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmimetype\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtext/x-python\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     45\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mname\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpython\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     46\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnbconvert_exporter\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpython\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     47\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpygments_lexer\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mipython3\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     48\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mversion\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m3.8.18\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m     49\u001b[0m   }\n\u001b[1;32m     50\u001b[0m  },\n\u001b[1;32m     51\u001b[0m  \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnbformat\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m4\u001b[39m,\n\u001b[1;32m     52\u001b[0m  \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnbformat_minor\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m5\u001b[39m\n\u001b[1;32m     53\u001b[0m }\n",
      "\u001b[0;31mNameError\u001b[0m: name 'null' is not defined"
     ]
    }
   ],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "Spyder Editor\n",
    "Author : Mwangi\n",
    "Purpose : this program imports mainfunction \n",
    "\"\"\"\n",
    "# note this will call print ('is this a main function example ?') first and \n",
    "# print (\"call mainfunction \") second\n",
    "import mainfunction\n",
    "print (\"call mainfunction \")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "de196b48-f429-4003-bd09-a44d6222fecf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
