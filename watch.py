import watcher

config = {
    "proc":watcher.PROCESS,
    "trigger":["python","main.py"],
    "mode":0,
    "path":"./",
    "files":[
        "main.py"
    ],
}

watch = watcher.Watcher(**config)
watch.start()
watch.observe()