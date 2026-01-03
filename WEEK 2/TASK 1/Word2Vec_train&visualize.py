from nltk import sent_tokenize
import gensim
from gensim.utils import simple_preprocess
txt='''Pollution is a pressing global issue that threatens the health of our planet and all its inhabitants. It manifests in various forms, including air pollution, water pollution, and land pollution, each with devastating consequences for the environment and human well-being. Air pollution, primarily caused by industrial emissions, vehicular exhaust, and agricultural activities, poses a significant threat to public health. Harmful pollutants such as particulate matter, nitrogen oxides, sulfur dioxide, and volatile organic compounds contaminate the air we breathe, leading to respiratory diseases, cardiovascular problems, and even premature death. Furthermore, air pollution exacerbates climate change by contributing to the greenhouse effect and global warming, with far-reaching ecological implications. Water pollution, resulting from industrial discharge, sewage runoff, and improper waste disposal, poses a grave threat to aquatic ecosystems and human communities reliant on clean water sources. Contaminants such as heavy metals, pesticides, plastics, and pathogens degrade water quality, rendering it unfit for consumption and harming marine life. Additionally, polluted water bodies pose risks of disease transmission and ecosystem collapse, jeopardizing biodiversity and human livelihoods. Land pollution, stemming from indiscriminate dumping of waste, industrial activities, and improper land use practices, degrades soil fertility and disrupts ecosystems. Toxic chemicals, plastics, and hazardous waste leach into the soil, contaminating groundwater and posing risks to human health and agricultural productivity. Land pollution also contributes to habitat destruction, loss of biodiversity, and soil erosion, exacerbating environmental degradation and undermining the sustainability of ecosystems. Addressing pollution requires concerted efforts at the individual, community, national, and international levels. Adopting sustainable practices such as reducing energy consumption, minimizing waste generation, and promoting eco-friendly alternatives can help mitigate pollution's adverse effects. Implementing stringent environmental regulations, investing in pollution control technologies, and fostering public awareness and education are essential steps toward combating pollution and safeguarding the planet for future generations. In conclusion, pollution poses a grave threat to environmental sustainability, human health, and ecosystem integrity. Addressing this multifaceted problem requires collective action and commitment to implementing effective solutions. By adopting sustainable practices, embracing renewable energy sources, and prioritizing environmental conservation, we can mitigate pollution's harmful effects and build a cleaner, healthier future for all.'''
txt = txt.lower()
sentences = sent_tokenize(txt)

tokenized_sentences =[simple_preprocess(sentence) for sentence in sentences]

model = gensim.models.Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=10,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100
       
)
y = model.wv.index_to_key

total_unique = len(set(word for sentence in tokenized_sentences for word in sentence))
print("Total unique words:", total_unique)
print("Words learned by Word2Vec Model:", len(y))
choice = input("IF YOU WANT TO VISUALIZE THE WORD EMBEDDINGS ENTER : 1\nIF YOU WANT TO GET VECTOR OF ANY WORD ENTER : 2\nENTER YOUR CHOICE HERE---- ")
if choice=='1':
    from sklearn.decomposition import PCA
    pca =PCA(n_components=3)
    X= pca.fit_transform(model.wv.get_normed_vectors())
    import plotly.express as px
    fig = px.scatter_3d(X[:],x=0,y=1,z=2,color=y[:])
    fig.show()
elif choice=='2':
    word=input("ENTER THE WORD:")
    word=word.lower()
    if word in y:
        print(model.wv[word]) 
    else:
        print("SORRY THE WORD ",word," IS NOT PRESENT IN THE CORPUS PLEASE TRY ANOTHER WORD")
else:
    print("PLEASE SELECT THE NUMBERS CORRECTLY")