from database import *
from sqlalchemy.orm import sessionmaker
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from datetime import datetime
from sqlalchemy import and_

if __name__ == "__main__":
    engine = create_engine('postgresql://postgres:marzipan@localhost/airInfoDB', echo=True)
    meta = MetaData(schema='warehouse')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    factors1 = ['month', 'dimairlinesid', 'scheduleddep', 'originairportid', 'destairportid',
                'scheduledtime', 'distance', 'scheduledarrival', 'arrivaltime', 'diverted', 'arrivaldelay']
    query = session.query(Flight).filter(and_(Flight.id <= 300000, Flight.id % 3 == 0))
    factorsArrays = [[] for _ in range(11)]
    for i in query.all():
        queryMonth = session.query(Date).filter(Date.id == i.dimdatesid).one()
        factorsArrays[0].append(queryMonth.month)
        for j in range(1, 11):
            eval(f"factorsArrays[{j}].append(i.{factors1[j]})")

    df = pd.DataFrame(factorsArrays)
    df = df.T
    df.columns = factors1
    print(df)

    labels = sorted(list(pd.unique(df['dimairlinesid'])))
    labelsTextQuery = session.query(Airline)
    labelsText = [labelsTextQuery.filter(Airline.id == i).one().airlinename for i in [int(k) for k in list(labels)]]
    print(labelsText)

    divertedCount = df[df['diverted'] == True]
    print(f'Diverted = {divertedCount.diverted.count()} out of {df.diverted.count()}')

    df = df.fillna(df.mean())
    print(df)


    def getStats(group):
        return {'min': group.min(), 'max': group.max(),
                'count': group.count(), 'mean': group.mean()}


    globalStats = df['arrivaldelay'].groupby(df['dimairlinesid']).apply(getStats).unstack()
    globalStats = globalStats.sort_index()
    globalStats.index = labelsText
    globalStats = globalStats.sort_values('count')
    print(globalStats)

    globalStats.plot(y='count', kind='bar')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(df.groupby('dimairlinesid').mean()['arrivaldelay'], labels=labels,
                                      autopct='%1.1f%%',
                                      textprops=dict(color='w'), colors=["#025464", "#E57C23", "#F266AB", "#2CD3E1",
                                                                         "#643843", "#1B9C85", "#4C4C6D", "#0C134F",
                                                                         "#99A98F", "#B04759", "#FEFF86", "#212A3E",
                                                                         "#9384D1", "#263A29", "#B2A4FF", "#000000"])

    ax.set_title('Average delay by airline')
    legend = ax.legend(wedges, labelsText,
                       title='Airline',
                       loc='center left',
                       bbox_to_anchor=(1, 0, 0, 1))
    plt.setp(autotexts, size=12, weight='bold')
    plt.show()
    fig.savefig('mean-piechart', bbox_extra_artists=(legend,), bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(df.groupby('dimairlinesid').count()['arrivaldelay'], labels=labels,
                                      autopct='%1.1f%%',
                                      textprops=dict(color='w'), colors=["#025464", "#E57C23", "#F266AB", "#2CD3E1",
                                                                         "#643843", "#1B9C85", "#4C4C6D", "#0C134F",
                                                                         "#99A98F", "#B04759", "#FEFF86", "#212A3E",
                                                                         "#9384D1", "#263A29", "#B2A4FF", "#000000"])

    ax.set_title('Number of flights by airline')
    legend = ax.legend(wedges, labelsText,
                       title='Airline',
                       loc='center left',
                       bbox_to_anchor=(1, 0, 0, 1))
    plt.setp(autotexts, size=12, weight='bold')
    plt.show()
    fig.savefig('count-piechart', bbox_extra_artists=(legend,), bbox_inches='tight')

    sns.jointplot(x=df.scheduledarrival, y=df.arrivaltime)
    plt.show()

    corr = df.corr()
    sns.heatmap(corr)
    plt.show()

    result = []
    for row in df.arrivaldelay:
        if row > 15:
            result.append(1)
        else:
            result.append(0)

    df['result'] = result
    print(df)
    print(df.value_counts('result'))

    data = df.values
    X, y = data[:, :-4], data[:, -1]
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=15)
    standardizer = StandardScaler()
    X_train, X_test = standardizer.fit_transform(X_train), standardizer.fit_transform(X_test)

    predResults = []
    clf = DecisionTreeClassifier()
    time1 = datetime.now()
    clf = clf.fit(X_train, y_train)
    time2 = datetime.now()
    print(f'Decision Tree training time = {time2 - time1}')
    pred = clf.predict(X_test)
    time1 = datetime.now()
    print(f'Decision Tree evaluation for the training set = {clf.score(X_train, y_train):.2%}')
    predResults.append(clf.score(X_test, y_test)*100)
    print(f'Decision Tree evaluation for the test set = {predResults[0]:.2%}')
    print(f'Decision Tree prediction time = {time1 - time2}')
    sns.heatmap(confusion_matrix(y_test, pred), annot=True)
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])
    search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(X_train, y_train)
    best = classifier.best_estimator_.get_params()["knn__n_neighbors"]
    print("\nThe best K for KNN =", best)

    time1 = datetime.now()
    knn = KNeighborsClassifier(n_neighbors=best, n_jobs=-1).fit(X_train, y_train)
    time2 = datetime.now()
    print(f'KNN training time = {time2 - time1}')
    pred = knn.predict(X_test)
    time1 = datetime.now()
    print(f'KNN evaluation for the training set = {knn.score(X_train, y_train):.2%}')
    predResults.append(knn.score(X_test, y_test)*100)
    print(f'KNN evaluation for the test set = {predResults[1]:.2%}')
    print(f'KNN prediction time = {time1 - time2}')
    sns.heatmap(confusion_matrix(y_test, pred), annot=True)
    plt.show()

    time1 = datetime.now()
    svc = SVC(kernel="rbf").fit(X_train, y_train)
    time2 = datetime.now()
    print(f'SVC training time = {time2 - time1}')
    pred = svc.predict(X_test)
    time1 = datetime.now()
    print(f'\nSVC evaluation for the training set = {svc.score(X_train, y_train):.2%}')
    predResults.append(svc.score(X_test, y_test)*100)
    print(f'SVC evaluation for the test set = {predResults[2]:.2%}')
    print(f'SVC prediction time = {time1 - time2}')
    sns.heatmap(confusion_matrix(y_test, pred), annot=True)
    plt.show()

    x_st = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=2, max_iter=500, algorithm="lloyd", random_state=1)
    kmeans.fit(x_st)
    time1 = datetime.now()
    predictKmeans = kmeans.predict(x_st)
    time2 = datetime.now()
    sns.heatmap(confusion_matrix(y, predictKmeans), annot=True)
    plt.show()

    ck = 0
    for i in range(len(y)):
        if y[i] == predictKmeans[i]:
            ck += 1
    print(f"\nK-Means Clustering evaluation = {ck / len(y) * 100}%")
    print(f'K-Means prediction time = {time2 - time1}')

    predResults.append(ck / len(y) * 100)
    predResults = pd.Series(predResults, index=["Decision Tree", "KNN", "SVC", "K-Means"], name="Score")
    predResults.plot(y="Score",kind="bar")
    plt.ylim([0,100])
    plt.show()