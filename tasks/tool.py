from invoke import task

@task(default=True)
def classifier(context):
    from _utils.gui import App

    app = App()
    app.mainloop()
