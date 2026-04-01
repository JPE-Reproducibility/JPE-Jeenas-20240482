# JPE Report Template 

> This template is heavily based on the AEA Data Editor's template https://github.com/AEADataEditor/replication-template 
> Thanks! 🙏

## Required Tools to use Template

1. Install [Quarto](https://quarto.org/docs/get-started/) to compile the report. 
   1. Quarto works more like latex and less then Word: you need to include all inputs to the document (like pictures) if sharing with others, or those elements will not appear.
   2. Given that *sharing* here works via `git`, you will have to `commit` all the files I need to compile your report.
2. `git` 
3. `VScode`: you should install the [markdown paste](https://marketplace.visualstudio.com/items/?itemName=telesoho.vscode-markdown-paste-image) extension to easily paste screenshots into the `.qmd` file.

## Outline

* clone this repo to a findable location on your computer (maybe `~/JPE-replications`?). Click on the green button "Code" and copy the URL. Do this
   ```
   cd ~/JPE-replications # or wherever you prefer
   git clone https://github.com/JPE-Reproducibility/JPE-Oswald-66666666.git
   ```
* If so, you will have `~/JPE-replications/JPE-Oswald-66666666` on your machine. You will work inside this folder.
* The content of this repo right now is:

```
.
├── LICENSE
├── README.md
├── TEMPLATE.qmd
├── _quarto.yml
├── _variables.yml
├── generated
├── images
├── package-output-map.xlsx
├── paper-appendices
├── replication-package <- code only
└── tools
```

Few remarks:

1. You will edit the file `TEMPLATE.qmd` in your text editor (VScode ideally). 
2. Your screenshots will be saved to `images/` if you use VScode with the correct extension (see below)
3. The files `_quarto.yml` and `_variables.yml` are generated inputs for the template, as is the content of the `generated/` folder.
4. the folder `replication-package` contains everything which considered *code* of some sort, so we track it here. You need to overwrite this folder with the **full unzipped package**, such that you have the full package content (the data, not only the code!). You will see that the folder structure is identical to what you see in the full package - I just omitted from here everything that is not code. See next step.
5. `tools` contains useful config files which you can use to prepare a clean environment.
6. the folder `paper-appendices` is available via the shared dropbox link in your assignment email.


## Next Step

* In your assignment email, there was a link to a dropox folder. Download and unzip the replication package as `replication-package` **into this repository**. That is, your repo should look like this after you downloaded the package:

```
├── LICENSE
├── README.md
├── TEMPLATE.html
├── TEMPLATE.qmd
├── TEMPLATE_files
├── _quarto.yml
├── _variables.yml
├── generated
├── images
├── package-output-map.xlsx
├── paper-appendices     <- copied from dropbox
├── replication-package  <- code, data, docs, everything
└── tools  

```

**Do NOT copy `replication-package` from dropbox to another location outside this repository!**



## Recommended workflow

1. Get VScode
2. install [markdown paste](https://marketplace.visualstudio.com/items/?itemName=telesoho.vscode-markdown-paste-image) extension
3. Go to [https://jpe-reproducibility.github.io/jpereplicators/](https://jpe-reproducibility.github.io/jpereplicators/) for step by step guidance.


