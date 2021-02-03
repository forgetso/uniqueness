import React, {Component} from 'react';
import './App.css';
import {IMaskInput} from 'react-imask';
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
    {"type": "hyponym", "label": "an example of"},
    {"type": "meronym", "label": "a part of"},
    //{"type": "holonym", "label": "a sibling of"},
];

class WordAssociation extends Component {
    constructor(props) {
        super(props);
        this.state = {
            relation_type: '',
            relations: [],
            cue: '',
            time: 0,
            showInput: false,
        };
        this.reload = this.reload.bind(this);
    }


    componentDidMount() {
        api.getRelationQuery().then(cue_obj => {
            console.log(cue_obj);
            this.setState({
                cue: cue_obj['word'],
                relations: cue_obj['relations'],
                relation_type: [RelationsTypes[Math.floor(Math.random() * RelationsTypes.length)]][0],
                time_start: Date.now(),
                showInput: true
            });
            this.getFingerprint();
            //this.textInput.getInputDOMNode().focus();
        }).catch(err => console.log('There was an error:' + err));
    }

    mySubmitHandler = (event) => {
        event.preventDefault();
        let relations = this.state.relations;
        let cue = this.state.cue;
        let relation_type = this.state.relation_type;
        let time_duration = Date.now() - this.state.time_start;
        let fingerprint = ls.get('fingerprint');
        let checkedItems = new Map();
        relations.map(word => checkedItems.set(word[0], event.target.elements[word[0]].checked * word[1]));
        this.setState({showInput: false});
        if (checkedItems.size > 0) {
            console.log(checkedItems);
            let checkedObj = Object.fromEntries(checkedItems);
            console.log(checkedObj);
            console.log(relation_type);
            api.saveResponse(cue, checkedObj, relation_type["type"], time_duration, fingerprint).then((res) => {
                if (res.status === 201) {
                    return res.json();
                }
            }).then((data) => {
                console.log(data);
                this.setState({'response_id': data['response_id']})
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
                                                            relation_type={this.state.relation_type}
                                                            relations={this.state.relations}>
                                </RelationsCheckboxComponent>

                                <input type="submit" value="next"/>
                            </div>
                            : <div class="spinner-border" role="status">
                                <span class="sr-only">Loading...</span>
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
        let checkedItems = new Map();
        let words = this.props.relations.map(x => x[0]);
        words.map(word => checkedItems.set(word, false));
        this.state = {checkedItems: checkedItems};
        this.MakeCheckbox = this.MakeCheckbox.bind(this);
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange = (event) => {
        const item = event.target.name;
        const isChecked = event.target.checked;
        this.setState(prevState => ({checkedItems: prevState.checkedItems.set(item, isChecked)}));
    };

    MakeCheckbox(wordScore, key) {
        let word = wordScore[0];
        let score = wordScore[1];
        return <div className="checkbox" key={key}>
            <label>
                <input type="checkbox" id={word} name={word} value={score} className="btn btn-primary"
                       checked={this.state.checkedItems.get(word)} onChange={this.handleChange}/>
                {word}
            </label>
        </div>;
    }

    render() {
        return (
            <div className="container text-center">
                <div>{this.props.relation_type["label"]}</div>
                <div className="container-sm">{this.props.relations.map(this.MakeCheckbox)}</div>
                <div className="small">Check all that apply</div>
            </div>
        )
    }
}

export default App;
