import React, {Component} from 'react';
import './App.css';
import Fingerprint2 from 'fingerprintjs2'
import ls from 'local-storage'
import * as api from "./helpers/api.js"

class App extends Component {
    render() {
        return (
            <div className="App">
                <WordAssociation></WordAssociation>
            </div>
        );
    }
}

//TODO prevent form submission when field is blank
//TODO allow spaces in form submission
let RelationsTypes = [
    //{"type": "hyponym", "label": "an example of"},
    {"type": "meronym", "label": "a part of"},
    //{"type": "holonym", "label": "a sibling of"},
];

class WordAssociation extends Component {
    constructor(props) {
        super(props);
        this.state = {
            relationType: '',
            relations: [],
            word_data: [],
            cue: '',
            time: 0,
            showInput: false,
            checkedItems: {}
        };
        this.reload = this.reload.bind(this);
    }

    componentDidMount() {
        api.getRelationQuery().then(resp => {
            console.log(resp);
            let checkedItems = new Map();
            let wordids = resp['data'].map(x => x['relation']);
            wordids.map(wordids => checkedItems.set(wordids, {'checked': false, 'time': 0, 'id': ''}));
            console.log(wordids);
            this.setState({
                wordData: resp['data'],
                cue: resp['data'][0]['cue'],
                relationType: [RelationsTypes[Math.floor(Math.random() * RelationsTypes.length)]][0],
                timeStart: Date.now(),
                showInput: true,
                checkedItems: checkedItems
            });
            this.getFingerprint();
            //this.textInput.getInputDOMNode().focus();
        }).catch(err => console.log('There was an error:' + err));
    }


    mySubmitHandler = (event) => {
        event.preventDefault();
        let relationType = this.state.relationType['type'];
        let fingerprint = ls.get('fingerprint');
        let checkedItems = this.state.checkedItems;
        let relationGroupId = this.state.wordData[0]['relationgroupid'];
        this.setState({showInput: false});
        for (const [key, _] of checkedItems) {
            if (!checkedItems.get(key)['checked']) checkedItems.delete(key);
        }
        if (checkedItems.size > 0) {
            console.log(checkedItems);
            console.log(Array.from(checkedItems.values()).map(item => item['time']));
            let relationIds = Array.from(checkedItems.values()).map(item => item['id']);
            let timeDurations = Array.from(checkedItems.values()).map(item => item['time']);
            api.saveResponse(relationGroupId, relationIds, timeDurations, fingerprint, relationType).then((res) => {
                if (res.status === 201) {
                    return res.json();
                }
            }).then((data) => {
                console.log(data);
                this.reload()
            });
        } else {
            this.reload()
        }
    };


    reload() {
        this.componentDidMount();
    }

    getFingerprint() {
        if (window.requestIdleCallback) {
            requestIdleCallback(this.idleFingerprint.bind(this));
        } else {
            setTimeout(this.idleFingerprint.bind(this), 500)
        }
    }

    idleFingerprint() {
        Fingerprint2.get(function (components, result) {
            //console.log(components); // an array of components: {key: ..., value: ...}
            var values = components.map(function (component) {
                return component.value
            });
            var murmur = Fingerprint2.x64hash128(values.join(''), 31);
            ls.set('fingerprint', murmur)
        })
    }


    render() {
        return (
            <form onSubmit={this.mySubmitHandler} className="display-4">
                <div>
                    <div>
                        {this.state.showInput ?
                            <div>
                                <span id="cue" className="m-2"><b><i>{this.state.cue}</i></b> is</span><br/>

                                <RelationsCheckboxComponent id="relations" className="m-2"
                                                            relationType={this.state.relationType}
                                                            timeStart={this.state.timeStart}
                                                            wordData={this.state.wordData}
                                                            checkedItems={this.state.checkedItems}>
                                </RelationsCheckboxComponent>

                                <input type="submit" value="next"/>
                            </div>
                            : <div className="spinner-border" role="status">
                                <span className="sr-only">Loading...</span>
                            </div>}
                    </div>
                </div>
            </form>
        );
    }
}


class RelationsCheckboxComponent extends React.Component {
    constructor(props) {
        super(props);
        this.props = props;
        this.state = {checkedItems: this.props.checkedItems};
        this.MakeCheckbox = this.MakeCheckbox.bind(this);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange = (event) => {
        const name = event.target.name;
        console.log(name);
        const isChecked = event.target.checked;
        let timeDuration = (Date.now() - this.props.timeStart) * isChecked;
        console.log(isChecked);
        let newValue = {'checked': isChecked, 'time': timeDuration, 'id': event.target.value};
        this.setState(prevState => ({checkedItems: prevState.checkedItems.set(name, newValue)}));
        console.log(this.state.checkedItems);

        //this.setState({event.target.id: time_duration})
    };

    MakeCheckbox(relationData, key) {
        let word = relationData['relation'];
        let relationid = relationData['relationid'].toString();
        return <div className="checkbox" key={key}>
            <label>
                <input type="checkbox" id={relationid} name={word} value={relationid} className="btn btn-primary"
                       checked={this.state.checkedItems.get(word)['checked']} onChange={this.handleChange}/>
                {word}
            </label>
        </div>;
    }

    render() {
        return (
            <div className="container text-center">
                <div>{this.props.relationType["label"]}</div>
                <div className="container-sm">{this.props.wordData.map(this.MakeCheckbox)}</div>
                <div className="small">Check all that apply</div>
            </div>
        )
    }
}

export default App;
