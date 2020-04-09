'use strict';
import React, {Component} from 'react';

class CorrectionsList extends Component {
    constructor(props) {
        super(props);
        this.state = {};
    }

    componentDidMount() {
        console.log(this.props.corrections);
        let count = 0;
        let word_array = Object.keys(this.props.corrections).sort();
        console.log(word_array);
        this.setState({word_array: word_array})
        for (const word of word_array) {
            if (count === 0) {
                this.setState({[word + '_show']: true});

            } else {
                this.setState({[word + '_show']: false});
            }
            this.setState({[word + '_submitted']: false});
            count++;
        }
        this.setState({'showSpelling': true})

    }

    unorderedList = function () {
        let word_array = this.state.word_array;
        return (
            word_array.map((word) =>

                this.state[word + '_show'] ?
                    <div key={word}>
                        <span className="display-4">By {word}, did you mean one of the following?</span>
                        <ul className="list-group display-2">
                            {this.listItems(word, this.props.corrections[word], word_array.indexOf(word))}
                            <li key="none_of_the_above">
                                <button type="button" className="btn btn-secondary btn-lg btn-block"
                                        onClick={(event) => this.hideCorrections(word)}>None of the above
                                </button>
                            </li>
                        </ul>
                    </div>
                    : null)
        )
    };

    listItems = function (original_word, corrections, index) {
        return (
            corrections.map((correction, index) =>
                <li key={correction.toString()}>
                    <button type="button" className="btn btn-primary btn-lg btn-block"
                            onClick={(event) => this.handleClick(event, original_word, correction, index)}>{correction}</button>
                </li>
            )
        )
    };

    handleClick = (event, original_word, correction, index) => {
        this.props.sendSpellingCorrection(this.props.response_id, original_word, correction);
        this.setState({[original_word + '_show']: false});
        this.setState({[original_word + '_submitted']: true});
        let next_index = this.state.word_array.indexOf(original_word) + 1
        console.log(next_index);
        let next_word = this.state.word_array[next_index];
        console.log('settting ' + next_word + ' to true');
        this.setState({[next_word + '_show']: true});
    };

    hideCorrections = (word) => {
        this.setState({[word + '_show']: false});
        this.setState({[word + '_submitted']: true});
        for (const word of this.state.word_array) {
            if (!this.state[word + '_submitted']) {
                console.log('rendering');

            } else {
                this.props.reloadComponent();
            }
        }
    };

    render() {
        //make sure the state has the first key at least
        return this.state.showSpelling ? (
            <div>
                {this.unorderedList()}
            </div>
        ) : null
    }
}


export default CorrectionsList;
