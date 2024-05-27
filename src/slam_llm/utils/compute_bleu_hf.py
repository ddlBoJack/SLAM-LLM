import evaluate


predictions = ["人工智能模型需要解释可解释的。", "罗伯特·哈蒙德写信给奥利弗·克伦威尔，请求他爱情而做恕处理。", "我将永远爱我的家人和朋友。", "这些例子主要来自英语世界。"]
references = [["人工智能模型需要是可解释的。"], ["罗伯特·哈蒙德写信给奥利弗·克伦威尔，请求为了爱可以宽大处理。"], ["我将永远爱我的家人和朋友。"], ["这里的例子主要来自英语世界。"]]

sacrebleu = evaluate.load("sacrebleu")
results = sacrebleu.compute(predictions=predictions, 
                             references=references,tokenize="zh")
print(results)
print(list(results.keys()))
print(round(results["score"], 1))